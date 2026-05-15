mod chess;
mod engine;

use chess::State;

fn main() {
    let mut state = State::new();
    let start_time = std::time::Instant::now();

    let depth = 5;
    println!("Running perft depth {} on starting position...", depth);
    state.divide(depth);

    let elapsed = start_time.elapsed();
    println!("Time taken: {:?}", elapsed);
}
