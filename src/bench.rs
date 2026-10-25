use crate::{
    chess::State,
    engine::{
        mcts::{Mcts, MctsConfig},
        model::{TransformerModel, TransformerModelConfig},
    },
};
use burn::tensor::backend::Backend;
use std::time::{Duration, Instant};

const MIN_DURATION: Duration = Duration::from_secs(2);
const MIN_ITERS: u32 = 2;
const MAX_WARM_UP_ITERS: usize = 10;
// Warm-up ends when a call takes over 90% as long as the one before it
const SETTLED_RATIO: f64 = 0.9;
const SEARCH_N_SIMS: usize = 128;

pub(crate) fn run<B: Backend>(device: &B::Device, max_batch_size: usize) {
    let model = TransformerModelConfig::new().init::<B>(device);
    println!("Backend: {}", B::name(device));

    println!("\nNetwork: input encoding, forward pass and readback");
    println!("{:>6} {:>10} {:>12}", "batch", "ms/call", "positions/s");
    for batch_size in (0..)
        .map(|i| 4usize.pow(i))
        .take_while(|&size| size <= max_batch_size)
    {
        let per_call = time_per_iter(|| evaluate_batch(&model, batch_size, device));
        println!(
            "{:>6} {:>10.2} {:>12.0}",
            batch_size,
            per_call.as_secs_f64() * 1e3,
            batch_size as f64 / per_call.as_secs_f64()
        );
    }

    println!("\nSearch: {SEARCH_N_SIMS} simulations from the starting position");
    let mut mcts = Mcts::new(&model, MctsConfig::new().with_n_sims(SEARCH_N_SIMS), device);
    let mut state = State::new();
    let per_search = time_per_iter(|| {
        mcts.search(&mut state);
    });
    println!(
        "{:.0} simulations/s",
        SEARCH_N_SIMS as f64 / per_search.as_secs_f64()
    );
}

// The network's cost doesn't depend on the position, so one position fills the batch
fn evaluate_batch<B: Backend>(model: &TransformerModel<B>, batch_size: usize, device: &B::Device) {
    let state = State::new();
    let states = vec![&state; batch_size];
    let moves: Vec<_> = states.iter().map(|state| state.generate_moves()).collect();

    model.evaluate(&states, &moves, device);
}

/// Mean duration of `f` once its calls stop getting faster. GPU backends compile and autotune
/// kernels during the first calls with each new tensor shape, which can take seconds per call.
fn time_per_iter(mut f: impl FnMut()) -> Duration {
    let mut previous = time_once(&mut f);
    for _ in 0..MAX_WARM_UP_ITERS {
        let current = time_once(&mut f);
        let settled = current.as_secs_f64() > SETTLED_RATIO * previous.as_secs_f64();
        previous = current;
        if settled {
            break;
        }
    }

    let start = Instant::now();
    let mut iters = 0;
    while iters < MIN_ITERS || start.elapsed() < MIN_DURATION {
        f();
        iters += 1;
    }

    start.elapsed() / iters
}

fn time_once(f: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}
