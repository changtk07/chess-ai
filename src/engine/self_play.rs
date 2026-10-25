use crate::{
    chess::{moves::Move, types::Color, Outcome, State},
    engine::{
        mcts::{MctsConfig, Search},
        model::{Evaluation, TransformerModel},
        samples::Sample,
    },
};
use burn::{config::Config, tensor::backend::Backend};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};
use std::time::{Duration, Instant};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Config, Debug)]
pub(crate) struct SelfPlayConfig {
    /// Games played at once, whose positions are evaluated together in one batch
    #[config(default = 256)]
    n_parallel: usize,
    #[config(default = 256)]
    n_games: usize,
    /// Moves are sampled in proportion to visit counts for this many plies, then chosen greedily
    #[config(default = 30)]
    sampling_plies: usize,
    /// Games still going after this many plies are stopped
    #[config(default = 512)]
    max_plies: usize,
    /// Dirichlet noise mixed into the root priors of every search, so games explore moves the
    /// network rates low: concentration alpha, and the noise's share of each prior
    #[config(default = 0.3)]
    noise_alpha: f32,
    #[config(default = 0.25)]
    noise_fraction: f32,
    #[config(default = 0)]
    seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GameResult {
    WhiteWins,
    BlackWins,
    Draw,
    PlyLimit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GameRecord {
    pub(crate) result: GameResult,
    pub(crate) plies: usize,
}

#[derive(Clone, Default)]
pub(crate) struct SelfPlayStats {
    pub(crate) games: Vec<GameRecord>,
    /// Moves played and simulations run, including in games still going
    pub(crate) n_moves: usize,
    pub(crate) n_sims: usize,
    pub(crate) n_evaluations: usize,
    /// Batched evaluations, one per step
    pub(crate) n_steps: usize,
    pub(crate) elapsed: Duration,
    pub(crate) select_time: Duration,
    pub(crate) evaluate_time: Duration,
    pub(crate) apply_time: Duration,
}

struct Game {
    state: State,
    search: Search,
    // Each game has its own generator, so its moves don't depend on the other games
    rng: StdRng,
    ply: usize,
    // Simulations in the searches already finished this game
    n_sims: usize,
    // Positions played so far, waiting for the game's result
    pending: Vec<PendingSample>,
}

struct PendingSample {
    fen: String,
    visits: Vec<(Move, u16)>,
    turn: Color,
}

impl Game {
    fn new(mcts_config: &MctsConfig, seed: u64) -> Self {
        Self {
            state: State::new(),
            search: Search::new(mcts_config.clone()),
            rng: StdRng::seed_from_u64(seed),
            ply: 0,
            n_sims: 0,
            pending: vec![],
        }
    }

    fn restart(&mut self) {
        self.state = State::new();
        self.search.reset();
        self.ply = 0;
        self.n_sims = 0;
        self.pending.clear();
    }

    /// Plays the move of each completed search until a search needs the network, leaving its
    /// position pending. Returns the record and training samples instead if the game ends.
    fn advance(&mut self, config: &SelfPlayConfig) -> Option<(GameRecord, Vec<Sample>)> {
        while !self.search.next_leaf(&mut self.state) {
            self.n_sims += self.search.n_sims_done();
            let temperature = if self.ply < config.sampling_plies {
                1.0
            } else {
                0.0
            };
            let search_result = self.search.result();
            let mv = search_result
                .sample_move(temperature, &mut self.rng)
                .expect("only unfinished games are searched");

            self.pending.push(PendingSample {
                fen: self.state.to_fen(),
                visits: search_result
                    .visits
                    .iter()
                    .map(|&(mv, n_visits)| {
                        let n_visits =
                            u16::try_from(n_visits).expect("visit counts fit in 16 bits");
                        (mv, n_visits)
                    })
                    .collect(),
                turn: self.state.turn,
            });

            self.state.make_move(mv);
            self.ply += 1;

            if let Some(result) = self.result(config) {
                let record = GameRecord {
                    result,
                    plies: self.ply,
                };
                return Some((record, self.take_samples(result)));
            }

            self.search.reset();
        }

        None
    }

    fn apply(&mut self, mut evaluation: Evaluation, config: &SelfPlayConfig) {
        if self.search.leaf_is_root() && config.noise_fraction > 0.0 {
            add_noise(
                &mut evaluation.priors,
                config.noise_alpha,
                config.noise_fraction,
                &mut self.rng,
            );
        }
        self.search.apply(&mut self.state, evaluation);
    }

    // Gives each position played the game's result for its side to move
    fn take_samples(&mut self, result: GameResult) -> Vec<Sample> {
        let winner = match result {
            GameResult::WhiteWins => Some(Color::White),
            GameResult::BlackWins => Some(Color::Black),
            GameResult::Draw | GameResult::PlyLimit => None,
        };

        self.pending
            .drain(..)
            .map(|position| Sample {
                fen: position.fen,
                visits: position.visits,
                result: match winner {
                    None => 0,
                    Some(color) if color == position.turn => 1,
                    Some(_) => -1,
                },
            })
            .collect()
    }

    fn result(&self, config: &SelfPlayConfig) -> Option<GameResult> {
        match self.state.outcome(&self.state.generate_moves()) {
            Some(Outcome::Checkmate) => Some(match self.state.turn {
                Color::White => GameResult::BlackWins,
                Color::Black => GameResult::WhiteWins,
            }),
            Some(Outcome::Draw) => Some(GameResult::Draw),
            None => (self.ply >= config.max_plies).then_some(GameResult::PlyLimit),
        }
    }
}

/// Plays `n_games` games, `n_parallel` at a time. Each step, every game's search advances to a
/// position that needs the network, all of those positions are evaluated in one batch, and each
/// search takes in its result. `on_game` receives each finished game with its training samples,
/// and `on_progress` is called every few seconds. Games stopped at the ply limit count as draws.
///
/// The tree work runs on one thread. It takes about 2% of each step, and a thread pool measured
/// slower overall: dispatching thousands of microsecond tasks costs more than it saves, and even
/// the network calls got slower, likely from idle workers competing with the thread driving the GPU.
pub(crate) fn run<B: Backend>(
    model: &TransformerModel<B>,
    mcts_config: &MctsConfig,
    config: &SelfPlayConfig,
    device: &B::Device,
    mut on_game: impl FnMut(&GameRecord, Vec<Sample>),
    mut on_progress: impl FnMut(&SelfPlayStats),
) -> SelfPlayStats {
    let start = Instant::now();
    let mut last_progress = start;
    let mut stats = SelfPlayStats::default();

    let n_slots = config.n_parallel.min(config.n_games);
    let mut games: Vec<Game> = (0..n_slots)
        .map(|i| Game::new(mcts_config, config.seed + i as u64))
        .collect();
    let mut n_started = n_slots;
    let mut finished_sims = 0;

    loop {
        let select_start = Instant::now();
        let mut i = 0;
        while i < games.len() {
            let Some((record, samples)) = games[i].advance(config) else {
                i += 1;
                continue;
            };
            on_game(&record, samples);
            stats.games.push(record);
            finished_sims += games[i].n_sims;

            if n_started == config.n_games {
                games.remove(i);
                continue;
            }

            // The new game in this slot needs its first position evaluated too
            games[i].restart();
            n_started += 1;
            let record = games[i].advance(config);
            debug_assert!(
                record.is_none(),
                "a new game can't end before its first move"
            );
            i += 1;
        }
        stats.select_time += select_start.elapsed();

        if games.is_empty() {
            break;
        }

        let evaluate_start = Instant::now();
        let states: Vec<&State> = games.iter().map(|game| &game.state).collect();
        let moves: Vec<&[Move]> = games.iter().map(|game| game.search.leaf_moves()).collect();
        let evaluations = model.evaluate(&states, &moves, device);
        stats.evaluate_time += evaluate_start.elapsed();
        stats.n_evaluations += evaluations.len();
        stats.n_steps += 1;

        let apply_start = Instant::now();
        for (game, evaluation) in games.iter_mut().zip(evaluations) {
            game.apply(evaluation, config);
        }
        stats.apply_time += apply_start.elapsed();

        if last_progress.elapsed() >= PROGRESS_INTERVAL {
            update_totals(&mut stats, &games, finished_sims, start);
            on_progress(&stats);
            last_progress = Instant::now();
        }
    }

    update_totals(&mut stats, &games, finished_sims, start);
    stats
}

/// Mixes Dirichlet(alpha) noise into `priors`, which keep summing to 1. The noise is drawn by
/// normalizing independent Gamma(alpha, 1) samples.
fn add_noise(priors: &mut [f32], alpha: f32, fraction: f32, rng: &mut impl Rng) {
    let gamma = Gamma::new(alpha, 1.0).expect("the noise alpha must be positive");
    let noise: Vec<f32> = priors.iter().map(|_| gamma.sample(rng)).collect();
    let total: f32 = noise.iter().sum();
    if total <= 0.0 {
        return;
    }

    for (prior, noise) in priors.iter_mut().zip(noise) {
        *prior = (1.0 - fraction) * *prior + fraction * noise / total;
    }
}

fn update_totals(stats: &mut SelfPlayStats, games: &[Game], finished_sims: usize, start: Instant) {
    let finished_moves: usize = stats.games.iter().map(|record| record.plies).sum();
    let ongoing_moves: usize = games.iter().map(|game| game.ply).sum();
    stats.n_moves = finished_moves + ongoing_moves;

    let ongoing_sims: usize = games
        .iter()
        .map(|game| game.n_sims + game.search.n_sims_done())
        .sum();
    stats.n_sims = finished_sims + ongoing_sims;

    stats.elapsed = start.elapsed();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::TransformerModelConfig;
    use burn::backend::NdArray;

    fn tiny_model() -> TransformerModel<NdArray> {
        TransformerModelConfig::new()
            .with_n_blocks(1)
            .with_n_heads(2)
            .with_head_dimension(16)
            .init(&Default::default())
    }

    // Returns the stats and each finished game with its samples
    fn play(
        model: &TransformerModel<NdArray>,
        seed: u64,
    ) -> (SelfPlayStats, Vec<(GameRecord, Vec<Sample>)>) {
        let mcts_config = MctsConfig::new().with_n_sims(8);
        // More games than slots, so finished games are replaced and then slots retired
        let config = SelfPlayConfig::new()
            .with_n_parallel(4)
            .with_n_games(6)
            .with_max_plies(40)
            .with_seed(seed);
        let mut games = vec![];
        let stats = run(
            model,
            &mcts_config,
            &config,
            &Default::default(),
            |record, samples| games.push((record.clone(), samples)),
            |_| {},
        );
        (stats, games)
    }

    #[test]
    fn every_game_finishes() {
        let (stats, _) = play(&tiny_model(), 0);

        assert_eq!(stats.games.len(), 6);
        for record in &stats.games {
            assert!((1..=40).contains(&record.plies));
            // A game can also end by checkmate or a draw on the last allowed ply
            if record.result == GameResult::PlyLimit {
                assert_eq!(record.plies, 40);
            }
        }

        // Every move came from a complete search of 8 simulations
        let total_plies: usize = stats.games.iter().map(|record| record.plies).sum();
        assert_eq!(stats.n_moves, total_plies);
        assert_eq!(stats.n_sims, total_plies * 8);
        // Each step evaluates one position per game still going, never more than the slots
        assert!(stats.n_steps <= stats.n_evaluations);
        assert!(stats.n_evaluations <= stats.n_steps * 4);
    }

    #[test]
    fn noise_keeps_priors_a_distribution() {
        let uniform = vec![0.05; 20];
        let noisy = |seed| {
            let mut priors = uniform.clone();
            add_noise(&mut priors, 0.3, 0.25, &mut StdRng::seed_from_u64(seed));
            priors
        };

        let priors = noisy(1);
        assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // Each prior keeps at least its unmixed share
        assert!(priors.iter().all(|&p| p >= 0.75 * 0.05 - 1e-6));
        assert_ne!(priors, uniform);
        assert_eq!(priors, noisy(1));
        assert_ne!(priors, noisy(2));
    }

    #[test]
    fn noise_changes_the_search() {
        // One game's first search with uniform priors, so only the noise can favour a move
        let first_search_visits = |config: &SelfPlayConfig| {
            let mut game = Game::new(&MctsConfig::new().with_n_sims(50), 0);
            while game.search.next_leaf(&mut game.state) {
                let n_moves = game.search.leaf_moves().len();
                let evaluation = Evaluation {
                    priors: vec![1.0 / n_moves as f32; n_moves],
                    value: 0.0,
                };
                game.apply(evaluation, config);
            }
            game.search.result().visits
        };

        let noisy = first_search_visits(&SelfPlayConfig::new());
        assert_eq!(noisy, first_search_visits(&SelfPlayConfig::new()));
        assert_ne!(
            noisy,
            first_search_visits(&SelfPlayConfig::new().with_noise_fraction(0.0))
        );
    }

    #[test]
    fn every_move_becomes_a_sample() {
        let (_, games) = play(&tiny_model(), 0);

        for (record, samples) in &games {
            assert_eq!(samples.len(), record.plies);
            assert_eq!(samples[0].fen, State::new().to_fen());

            for sample in samples {
                // The search ran 8 simulations, and the first one expanded the root
                let total_visits: u16 = sample.visits.iter().map(|(_, n)| n).sum();
                assert_eq!(total_visits, 7);

                // The FEN is the searched position, and its legal moves are all recorded
                let state = State::from_fen(&sample.fen);
                assert_eq!(state.to_fen(), sample.fen);
                let moves: Vec<Move> = sample.visits.iter().map(|(mv, _)| *mv).collect();
                assert_eq!(moves, state.generate_moves().as_slice());
            }
        }
    }

    #[test]
    fn results_are_for_the_side_to_move() {
        let results = |result| {
            let mut game = Game::new(&MctsConfig::new(), 0);
            for turn in [Color::White, Color::Black, Color::White] {
                game.pending.push(PendingSample {
                    fen: String::new(),
                    visits: vec![],
                    turn,
                });
            }
            let samples = game.take_samples(result);
            samples
                .iter()
                .map(|sample| sample.result)
                .collect::<Vec<_>>()
        };

        assert_eq!(results(GameResult::WhiteWins), [1, -1, 1]);
        assert_eq!(results(GameResult::BlackWins), [-1, 1, -1]);
        assert_eq!(results(GameResult::Draw), [0, 0, 0]);
        assert_eq!(results(GameResult::PlyLimit), [0, 0, 0]);
    }

    #[test]
    fn runs_with_the_same_seed_match() {
        let model = tiny_model();
        assert_eq!(play(&model, 7).1, play(&model, 7).1);
    }
}
