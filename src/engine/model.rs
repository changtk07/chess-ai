use crate::{
    chess::{moves::Move, State},
    engine::encoding::{NetworkInput, N_MOVE_PLANES, N_PIECE_TOKENS, N_STATE_FEATURES},
};
use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig,
    },
    tensor::{
        activation::{gelu, softmax, tanh},
        backend::Backend,
        Int, TensorData,
    },
    Tensor,
};

#[derive(Module, Debug)]
struct ChessEmbedding<B: Backend> {
    e_piece: Embedding<B>,
    e_pos: Embedding<B>,
    ff: Linear<B>,
}

impl<B: Backend> ChessEmbedding<B> {
    // [batch_size, 64] piece tokens and [batch_size, N_STATE_FEATURES] -> [batch_size, 65, d_model],
    // the CLS token first
    fn forward(&self, pieces: Tensor<B, 2, Int>, state_features: Tensor<B, 2>) -> Tensor<B, 3> {
        let squares = TensorData::new((0i32..64).collect(), [1, 64]);
        let squares = Tensor::from_data(squares, &pieces.device());

        let encoded_pieces = self.e_piece.forward(pieces);
        let encoded_squares = self.e_pos.forward(squares);
        let cls_token = self.ff.forward(state_features).unsqueeze_dim(1);

        Tensor::cat(vec![cls_token, encoded_pieces + encoded_squares], 1)
    }
}

#[derive(Config, Debug)]
struct ChessEmbeddingConfig {
    #[config(default = 384)]
    d_model: usize,
}

impl ChessEmbeddingConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ChessEmbedding<B> {
        ChessEmbedding {
            e_piece: EmbeddingConfig::new(N_PIECE_TOKENS, self.d_model).init(device),
            e_pos: EmbeddingConfig::new(64, self.d_model).init(device),
            ff: LinearConfig::new(N_STATE_FEATURES, self.d_model).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct AttentionBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: RmsNorm<B>,
    norm2: RmsNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> AttentionBlock<B> {
    // [batch_size, 65, d_model]
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm1.forward(input.clone());
        let x = self.attention.forward(MhaInput::self_attn(x)).context;
        let middle = input + x;
        let x = self.norm2.forward(middle.clone());
        let x = self.ff1.forward(x);
        let x = gelu(x);
        let x = self.ff2.forward(x);
        middle + x
    }
}

#[derive(Config, Debug)]
struct AttentionBlockConfig {
    #[config(default = 6)]
    n_heads: usize,
    #[config(default = 384)]
    d_model: usize,
    #[config(default = 768)]
    d_ff: usize,
}

impl AttentionBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> AttentionBlock<B> {
        AttentionBlock {
            attention: MultiHeadAttentionConfig::new(self.d_model, self.n_heads).init(device),
            norm1: RmsNormConfig::new(self.d_model).init(device),
            norm2: RmsNormConfig::new(self.d_model).init(device),
            ff1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff2: LinearConfig::new(self.d_ff, self.d_model).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct PolicyHead<B: Backend> {
    ff: Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    // [batch_size, 64, d_model] -> logits of the legal moves, [batch_size, max legal moves]
    fn forward(
        &self,
        input: Tensor<B, 3>,
        move_indices: Tensor<B, 2, Int>,
        padding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let x = self.ff.forward(input);
        let x = x.flatten(1, 2);
        x.gather(1, move_indices) + padding
    }
}

#[derive(Config, Debug)]
struct PolicyHeadConfig {
    #[config(default = 384)]
    d_model: usize,
}

impl PolicyHeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        PolicyHead {
            ff: LinearConfig::new(self.d_model, N_MOVE_PLANES).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct ValueHead<B: Backend> {
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
    // [batch_size, d_model] — CLS token
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.ff1.forward(input);
        let x = gelu(x);
        let x = self.ff2.forward(x);
        tanh(x)
    }
}

#[derive(Config, Debug)]
struct ValueHeadConfig {
    #[config(default = 384)]
    d_model: usize,
}

impl ValueHeadConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
        ValueHead {
            ff1: LinearConfig::new(self.d_model, self.d_model).init(device),
            ff2: LinearConfig::new(self.d_model, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct TransformerModel<B: Backend> {
    embedding: ChessEmbedding<B>,
    attention_blocks: Vec<AttentionBlock<B>>,
    policy_head: PolicyHead<B>,
    value_head: ValueHead<B>,
}

/// Network output for one position: a prior per legal move, in the order the moves were given,
/// and the value for the side to move.
pub(crate) struct Evaluation {
    pub(crate) priors: Vec<f32>,
    pub(crate) value: f32,
}

impl<B: Backend> TransformerModel<B> {
    /// Returns legal-move logits, [batch_size, max legal moves], and values, [batch_size, 1].
    /// Logits past a position's own move count are padding.
    pub(crate) fn forward(&self, input: NetworkInput<B>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let attn_input = self.embedding.forward(input.pieces, input.state_features);

        let attn_output = self
            .attention_blocks
            .iter()
            .fold(attn_input, |x, block| block.forward(x));

        let seq_len = attn_output.dims()[1];
        let cls_token = attn_output.clone().narrow(1, 0, 1).squeeze_dim(1); // [B, d_model]
        let squares = attn_output.narrow(1, 1, seq_len - 1); // [B, 64, d_model]

        let policy = self
            .policy_head
            .forward(squares, input.move_indices, input.padding);
        let value = self.value_head.forward(cls_token);

        (policy, value)
    }

    /// Evaluates a batch of positions, reading priors and values back from the device at once.
    pub(crate) fn evaluate(
        &self,
        states: &[&State],
        moves: &[impl AsRef<[Move]>],
        device: &B::Device,
    ) -> Vec<Evaluation> {
        let (logits, value) = self.forward(NetworkInput::new(states, moves, device));
        let width = logits.dims()[1];
        let output = Tensor::cat(vec![softmax(logits, 1), value], 1);
        let data = output.into_data().convert::<f32>().to_vec::<f32>().unwrap();

        data.chunks_exact(width + 1)
            .zip(moves)
            .map(|(row, moves)| Evaluation {
                priors: row[..moves.as_ref().len()].to_vec(),
                value: row[width],
            })
            .collect()
    }
}

#[derive(Config, Debug)]
pub(crate) struct TransformerModelConfig {
    #[config(default = 8)]
    n_blocks: usize,
    #[config(default = 6)]
    n_heads: usize,
    #[config(default = 2.0)]
    d_ff_scale: f64,
    #[config(default = 64)]
    head_dimension: usize,
}

impl TransformerModelConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> TransformerModel<B> {
        let d_model = self.n_heads * self.head_dimension;
        let d_ff = (self.d_ff_scale * d_model as f64) as usize;

        TransformerModel {
            embedding: ChessEmbeddingConfig::new()
                .with_d_model(d_model)
                .init(device),
            attention_blocks: (0..self.n_blocks)
                .map(|_| {
                    AttentionBlockConfig::new()
                        .with_n_heads(self.n_heads)
                        .with_d_model(d_model)
                        .with_d_ff(d_ff)
                        .init(device)
                })
                .collect(),
            policy_head: PolicyHeadConfig::new().with_d_model(d_model).init(device),
            value_head: ValueHeadConfig::new().with_d_model(d_model).init(device),
        }
    }
}

/// A one-block network that keeps CPU tests fast. Untrained, its outputs serve as well as any to
/// test the code around it.
#[cfg(test)]
pub(crate) fn tiny_model() -> TransformerModel<burn::backend::NdArray> {
    TransformerModelConfig::new()
        .with_n_blocks(1)
        .with_n_heads(2)
        .with_head_dimension(16)
        .init(&Default::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::state::START_FEN;
    use burn::{
        backend::{Cuda, NdArray},
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        tensor::{bf16, f16},
    };

    const FENS: [&str; 3] = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];

    fn evaluate_fens<B: Backend>(
        model: &TransformerModel<B>,
        fens: &[&str],
        device: &B::Device,
    ) -> Vec<Evaluation> {
        let states: Vec<State> = fens.iter().map(|fen| State::from_fen(fen)).collect();
        let states: Vec<&State> = states.iter().collect();
        let moves: Vec<_> = states.iter().map(|state| state.generate_moves()).collect();
        model.evaluate(&states, &moves, device)
    }

    // All priors in order, then all values
    fn flatten(evaluations: &[Evaluation]) -> (Vec<f32>, Vec<f32>) {
        let priors = evaluations.iter().flat_map(|e| e.priors.clone()).collect();
        let values = evaluations.iter().map(|e| e.value).collect();
        (priors, values)
    }

    // The positions have 20, 48 and 6 legal moves, so the shorter rows are padded
    #[test]
    fn batched_evaluation_matches_single_positions() {
        let model = tiny_model();
        let device = Default::default();
        let batched = evaluate_fens(&model, &FENS, &device);

        for (fen, evaluation) in FENS.iter().zip(&batched) {
            let single = evaluate_fens(&model, &[fen], &device).remove(0);
            let n_moves = State::from_fen(fen).generate_moves().len();

            assert_eq!(evaluation.priors.len(), n_moves);
            assert!((evaluation.priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(max_diff(&evaluation.priors, &single.priors) < 1e-5);
            assert!((evaluation.value - single.value).abs() < 1e-5);
        }
    }

    #[test]
    fn priors_follow_their_moves() {
        let model = tiny_model();
        let device = Default::default();
        let state = State::from_fen(FENS[1]);
        let moves = state.generate_moves();
        let reversed: Vec<Move> = moves.iter().rev().copied().collect();

        let forward = model.evaluate(&[&state], &[&moves[..]], &device).remove(0);
        let mut backward = model
            .evaluate(&[&state], &[&reversed[..]], &device)
            .remove(0);
        backward.priors.reverse();

        assert!(max_diff(&forward.priors, &backward.priors) < 1e-6);
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    // Runs the same weights on the CPU and on backend B
    fn assert_matches_cpu<B: Backend>(tolerance: f32) {
        let cpu_device = Default::default();
        let cpu_model = TransformerModelConfig::new().init::<NdArray>(&cpu_device);
        let (cpu_policy, cpu_value) = flatten(&evaluate_fens(&cpu_model, &FENS, &cpu_device));

        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder.record(cpu_model.into_record(), ()).unwrap();
        let device = Default::default();
        let model = TransformerModelConfig::new()
            .init::<B>(&device)
            .load_record(recorder.load(bytes, &device).unwrap());
        let (policy, value) = flatten(&evaluate_fens(&model, &FENS, &device));

        // `f32::max` skips NaN, so check for it before comparing
        assert!(policy.iter().chain(&value).all(|x| x.is_finite()));
        assert!(max_diff(&policy, &cpu_policy) < tolerance);
        assert!(max_diff(&value, &cpu_value) < tolerance);
    }

    // Tolerances are about 10x the differences measured on an RTX 4090
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_matches_cpu() {
        assert_matches_cpu::<Cuda>(1e-3);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_f16_matches_cpu() {
        assert_matches_cpu::<Cuda<f16>>(5e-3);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_bf16_matches_cpu() {
        assert_matches_cpu::<Cuda<bf16>>(3e-2);
    }
}
