mod bench;
mod chess;
mod engine;

use crate::{
    chess::{Outcome, State},
    engine::{
        mcts::{Mcts, MctsConfig},
        model::TransformerModelConfig,
    },
};
use burn::{
    backend::{Cuda, NdArray},
    tensor::{backend::Backend, bf16, f16},
};
use std::time::Instant;

const SELF_PLAY_N_SIMS: usize = 64;
const SELF_PLAY_MAX_PLIES: usize = 512;
// Sample moves early so repeated runs don't replay the same game
const SELF_PLAY_SAMPLING_PLIES: usize = 30;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let backend = take_option(&mut args, "--backend").unwrap_or_else(|| "cpu".into());

    match backend.as_str() {
        "cpu" => run::<NdArray>(&args, &Default::default()),
        "cuda" => run::<Cuda>(&args, &Default::default()),
        "cuda-f16" => run::<Cuda<f16>>(&args, &Default::default()),
        "cuda-bf16" => run::<Cuda<bf16>>(&args, &Default::default()),
        other => panic!("unknown backend {other}, expected cpu, cuda, cuda-f16 or cuda-bf16"),
    }
}

fn run<B: Backend>(args: &[String], device: &B::Device) {
    match args.first().map(String::as_str) {
        Some("perft") => perft(args.get(1).map_or(5, |depth| depth.parse().unwrap())),
        Some("bench") => bench::run::<B>(
            device,
            args.get(1).map_or(256, |size| size.parse().unwrap()),
        ),
        _ => self_play::<B>(device),
    }
}

// Removes `name` and the value after it from `args`, returning the value
fn take_option(args: &mut Vec<String>, name: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == name)?;
    args.remove(index);
    (index < args.len()).then(|| args.remove(index))
}

fn perft(depth: u8) {
    let mut state = State::new();
    let start_time = Instant::now();

    println!("Running perft depth {} on starting position...", depth);
    state.divide(depth);

    let elapsed = start_time.elapsed();
    println!("Time taken: {:?}", elapsed);
}

// Plays one game with untrained weights to exercise search end to end
fn self_play<B: Backend>(device: &B::Device) {
    let model = TransformerModelConfig::new().init::<B>(device);
    let config = MctsConfig::new().with_n_sims(SELF_PLAY_N_SIMS);
    let mut mcts = Mcts::new(&model, config, device);
    let mut rng = rand::rng();
    let mut state = State::new();
    let start_time = Instant::now();

    for ply in 0..SELF_PLAY_MAX_PLIES {
        let moves = state.generate_moves();

        if let Some(outcome) = state.outcome(&moves) {
            println!("\n{state}");
            match outcome {
                Outcome::Checkmate => println!("{} wins by checkmate", state.turn.flip()),
                Outcome::Draw => println!("Draw"),
            }
            println!("{} plies in {:?}", ply, start_time.elapsed());
            return;
        }

        let result = mcts.search(&mut state);
        let temperature = if ply < SELF_PLAY_SAMPLING_PLIES {
            1.0
        } else {
            0.0
        };
        let mv = result.sample_move(temperature, &mut rng).unwrap();

        println!(
            "{:>3}. {} {} (value {:+.2})",
            ply + 1,
            state.turn,
            mv,
            result.value
        );
        state.make_move(mv);
    }

    println!("\n{state}");
    println!(
        "Stopped after {SELF_PLAY_MAX_PLIES} plies in {:?}",
        start_time.elapsed()
    );
}
