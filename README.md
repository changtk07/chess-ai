# Cinder

Cinder is an AlphaZero-style chess engine in Rust: bitboard move generation, a transformer policy/value network built with [burn](https://burn.dev), and Monte Carlo tree search guided by the network.

The network isn't trained yet. The engine plays complete, legal games, but with random weights its moves are close to random.

## Usage

```sh
cargo run --release              # play one self-play game with untrained weights
cargo run --release -- perft 5   # count positions at depth 5 from the start, per root move
cargo run --release -- bench     # network throughput by batch size (up to 256), and search speed
cargo test                       # move generation, game outcome and search tests
```

Commands run on the CPU by default. Add `--backend cuda` to run on an NVIDIA GPU, or `cuda-f16` / `cuda-bf16` for half precision, e.g. `cargo run --release -- bench 4096 --backend cuda-f16`. The first run on a GPU spends a minute or more compiling kernels.

The CUDA backend needs the CUDA 13 toolkit, for its runtime compiler NVRTC. `.cargo/config.toml` pins cudarc to its CUDA 13.1 bindings, since cudarc 0.18 rejects newer 13.x toolkits. To check that the GPU computes the same results as the CPU:

```sh
cargo test --release -- --ignored
```

The wgpu backend is also enabled in `Cargo.toml`, but finds no Vulkan adapter under WSL2.

## How it works

**Chess core** (`src/chess/`)

- Bitboards, with attack tables and Zobrist keys generated at compile time.
- Legal move generation from pin and check masks, without trying moves and undoing them.
- Make/unmake with incremental Zobrist hashing.
- Game outcomes: checkmate, stalemate, the 50-move rule, threefold repetition and insufficient material.
- FEN parsing and perft.

**Network** (`src/engine/model.rs`)

- Input is 65 tokens: one per square (piece embedding plus square embedding), and a CLS token encoding side to move, castling rights, en passant file and the halfmove clock.
- 8 pre-norm transformer blocks: 6 heads, model width 384, RMSNorm, GELU.
- Policy head: 73 move planes per square, 4,672 moves in total, using the AlphaZero encoding in `src/engine/policy.rs`. Only the legal moves' logits are gathered on the device, so the softmax and the data read back cover just those moves.
- Value head: reads the CLS token and outputs a tanh score in [−1, 1] for the side to move.

**Search** (`src/engine/mcts.rs`)

- PUCT selection, with the network supplying move priors and position values at the leaves, and exact values for finished games.
- Each search returns root visit counts. The move is either the most visited or sampled with a temperature.

## Roadmap

- [ ] Batched leaf evaluation on a GPU backend, in progress: see [docs/performance-plan.md](docs/performance-plan.md)
- [ ] Self-play data generation
- [ ] Training loop
- [ ] UCI interface

## License

Apache 2.0. See [LICENSE](LICENSE).
