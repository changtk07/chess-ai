# Cinder

Cinder is an AlphaZero-style chess engine in Rust: bitboard move generation, a transformer policy/value network built with [burn](https://burn.dev), and Monte Carlo tree search guided by the network.

The network isn't trained yet. The engine plays complete, legal games, but with random weights its moves are close to random.

## Usage

```sh
cargo run --release -- self-play --backend cuda-bf16   # play 256 games at once with untrained weights
cargo run --release -- bench     # network throughput by batch size (up to 256), and search speed
cargo run --release -- perft 5   # count positions at depth 5 from the start, per root move
cargo test                       # move generation, game outcome, search and self-play tests
```

`self-play` takes `--parallel` (games at once, 256), `--sims` (simulations per move, 800), `--games` (total, 256) and `--seed`. Every few seconds it reports simulations and network evaluations per second, the average batch size, and how the time splits between tree search, the network, applying its results, and writing samples.

With `--out data/selfplay.bin`, each finished game's training samples are appended to that file: one per move played, with the position, every legal move's visit count from its search, and the game's result for the side to move. Games stopped at the ply limit count as draws. The format is described in `src/engine/samples.rs`.

Commands run on the CPU by default. Add `--backend cuda` to run on an NVIDIA GPU, or `cuda-f16` / `cuda-bf16` for half precision, e.g. `cargo run --release -- bench 4096 --backend cuda-f16`. The first run on a GPU spends a minute or more compiling kernels.

The CUDA backend needs the CUDA 13 toolkit, for its runtime compiler NVRTC. `.cargo/config.toml` pins cudarc to its CUDA 13.1 bindings, since cudarc 0.18 rejects newer 13.x toolkits. To check that the GPU computes the same results as the CPU:

```sh
cargo test --release -- --ignored
```

burn's wgpu backend isn't enabled: under WSL2 it finds no Vulkan adapter, so the GPU path is CUDA.

## How it works

**Chess core** (`src/chess/`)

- Bitboards, with attack tables and Zobrist keys generated at compile time.
- Legal move generation from pin and check masks, without trying moves and undoing them.
- Make/unmake with incremental Zobrist hashing.
- Game outcomes: checkmate, stalemate, the 50-move rule, threefold repetition and insufficient material.
- FEN parsing and perft.

**Network** (`src/engine/model.rs`)

- Input is 65 tokens: one per square (piece embedding plus square embedding), and a CLS token encoding castling rights, en passant file and the halfmove clock.
- The board is seen from the side to move: for black it's mirrored top to bottom with the colours swapped, and moves are mirrored to match. The network learns each pattern once rather than once per colour. Chess is the same game mirrored, so the side to move isn't an input, and a position and its mirror image evaluate identically.
- 8 pre-norm transformer blocks: 6 heads, model width 384, RMSNorm, GELU. A final RMSNorm normalizes the residual stream before the heads.
- Policy head: 73 move planes per square, 4,672 moves in total, using the AlphaZero encoding in `src/engine/encoding.rs`. Only the legal moves' logits are gathered on the device, so the softmax and the data read back cover just those moves.
- Value head: reads the CLS token and outputs a tanh score in [−1, 1] for the side to move.

**Search** (`src/engine/mcts.rs`)

- PUCT selection, with the network supplying move priors and position values at the leaves, and exact values for finished games.
- Each search returns root visit counts. The move is either the most visited or sampled with a temperature.
- A search pauses whenever it reaches a position the network must evaluate, so many searches can share one batch.

**Self-play** (`src/engine/self_play.rs`)

- Many games run at once. Each step, every game's search advances to its next position needing the network, and all of those positions are evaluated in one batch on the GPU.
- Moves are sampled by visit count for the first 30 plies, then chosen greedily. `--seed` fixes the untrained network's weights and seeds every game's own random generator, so runs with the same seed write the same samples, on the GPU as well.
- Dirichlet noise (alpha 0.3, a quarter of each prior) is mixed into every search's root priors, so games explore moves the network rates low.
- Every move played becomes a training sample, written out when its game ends.

## Roadmap

- [x] Batched leaf evaluation on a GPU backend: see [docs/performance.md](docs/performance.md)
- [x] Self-play data: Dirichlet noise at the root, and recording (position, visit distribution, result) samples
- [ ] Network changes, all before any weights are trained:
  - Move the network's input encoding next to the move encoding in `policy.rs`, so the board flip changes both consistently and training can encode samples without a forward pass.
  - Flip the board so the side to move is always at the bottom: the network then learns each pattern once instead of once per colour.
  - Add a final `RmsNorm` before the policy and value heads. The blocks normalize their inputs, so the residual stream otherwise reaches the heads unnormalized.
  - Set attention dropout deliberately: burn defaults it to 0.1, which only takes effect once training runs.
- [ ] Faster network, now about 98% of self-play time:
  - Flatten inputs to 2D before linear layers: burn's 3D form runs at about half the speed of a 2D matrix multiply.
  - Look at attention, and consider a smaller network.
- [ ] Reuse the search tree between moves:
  - First measure how many of each search's visits the chosen move's subtree holds: that's the share of evaluations reuse would save.
  - A reused root is already expanded, so noise must be mixed into its existing priors. Decide whether `--sims` means new simulations per move or total visits at the root, since samples record visits.
  - Caching evaluations by Zobrist hash and halfmove clock is an alternative or a complement.
- [ ] Training loop on the GPU
- [ ] Once the network is trained, score unvisited moves with first-play urgency (the parent's value minus a reduction, as Leela does) instead of as draws.
- [ ] UCI interface:
  - Print promotions as `a7a8q` rather than `a7a8=q`, together with the `play()` test helper, which matches moves by that string.
  - Validate the positions it's given: `State::from_fen` accepts impossible positions, where the side not to move is in check, and `Position::from_fen` panics on malformed squares.
- [ ] Cleanups, when convenient:
  - Merge the four promotion helpers and the two castling helpers in `State::make_move`, checked with perft.
  - Share one ray-attack helper between the rook and bishop attack masks and the pin and check masks, checking that perft 5 isn't slower.
  - Shrink search tree nodes from 40 to 24 bytes, with `u32` visit counts and a flag for finished games. Only needed for much larger `--parallel`.
  - Maybe set the en passant square only when a capture is possible. Today it's set after every double push, so a repetition can be detected up to a move late, and the network sees a flag that changes nothing.

## License

Apache 2.0. See [LICENSE](LICENSE).
