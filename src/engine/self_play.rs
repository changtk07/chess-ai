use crate::{
    chess::{moves::Move, types::Color, Outcome, State},
    engine::{
        mcts::{MctsConfig, Search},
        model::TransformerModel,
    },
};
use burn::{config::Config, tensor::backend::Backend};
use rand::{rngs::StdRng, SeedableRng};
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
}

impl Game {
    fn new(mcts_config: &MctsConfig, seed: u64) -> Self {
        Self {
            state: State::new(),
            search: Search::new(mcts_config.clone()),
            rng: StdRng::seed_from_u64(seed),
            ply: 0,
            n_sims: 0,
        }
    }

    fn restart(&mut self) {
        self.state = State::new();
        self.search.reset();
        self.ply = 0;
        self.n_sims = 0;
    }

    /// Plays the move of each completed search until a search needs the network, leaving its
    /// position pending. Returns the record instead if the game ends.
    fn advance(&mut self, config: &SelfPlayConfig) -> Option<GameRecord> {
        while !self.search.next_leaf(&mut self.state) {
            self.n_sims += self.search.n_sims_done();
            let temperature = if self.ply < config.sampling_plies {
                1.0
            } else {
                0.0
            };
            let mv = self
                .search
                .result()
                .sample_move(temperature, &mut self.rng)
                .expect("only unfinished games are searched");

            self.state.make_move(mv);
            self.ply += 1;

            if let Some(result) = self.result(config) {
                return Some(GameRecord {
                    result,
                    plies: self.ply,
                });
            }

            self.search.reset();
        }

        None
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
/// search takes in its result. `on_progress` is called every few seconds.
///
/// The tree work runs on one thread. It takes about 2% of each step, and a thread pool measured
/// slower overall: dispatching thousands of microsecond tasks costs more than it saves, and even
/// the network calls got slower, likely from idle workers competing with the thread driving the GPU.
pub(crate) fn run<B: Backend>(
    model: &TransformerModel<B>,
    mcts_config: &MctsConfig,
    config: &SelfPlayConfig,
    device: &B::Device,
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
            let Some(record) = games[i].advance(config) else {
                i += 1;
                continue;
            };
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
            game.search.apply(&mut game.state, evaluation);
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

    fn play(model: &TransformerModel<NdArray>, seed: u64) -> SelfPlayStats {
        let mcts_config = MctsConfig::new().with_n_sims(8);
        // More games than slots, so finished games are replaced and then slots retired
        let config = SelfPlayConfig::new()
            .with_n_parallel(4)
            .with_n_games(6)
            .with_max_plies(40)
            .with_seed(seed);
        run(model, &mcts_config, &config, &Default::default(), |_| {})
    }

    #[test]
    fn every_game_finishes() {
        let stats = play(&tiny_model(), 0);

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
    fn runs_with_the_same_seed_match() {
        let model = tiny_model();
        assert_eq!(play(&model, 7).games, play(&model, 7).games);
    }
}
