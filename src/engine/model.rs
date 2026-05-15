use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Linear, LinearConfig, RmsNorm, RmsNormConfig,
    },
    tensor::{
        activation::{gelu, softmax, tanh},
        backend::Backend,
    },
    Tensor,
};

#[derive(Module, Debug)]
struct AttentionBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm1: RmsNorm<B>,
    norm2: RmsNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> AttentionBlock<B> {
    // [batch_size, seq_len, d_model]
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
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    // [batch_size, d_model]
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.ff1.forward(input);
        let x = gelu(x);
        let x = self.ff2.forward(x);
        softmax(x, 1)
    }
}

#[derive(Config, Debug)]
struct PolicyHeadConfig {
    #[config(default = 384)]
    d_model: usize,
    #[config(default = 384)]
    d_ff: usize,
}

impl PolicyHeadConfig {
    const D_OUT: usize = 4672;

    fn init<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        PolicyHead {
            ff1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff2: LinearConfig::new(self.d_ff, Self::D_OUT).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct ValueHead<B: Backend> {
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
    // [batch_size, d_model]
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
    #[config(default = 384)]
    d_ff: usize,
}

impl ValueHeadConfig {
    const D_OUT: usize = 1;

    fn init<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
        ValueHead {
            ff1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff2: LinearConfig::new(self.d_ff, Self::D_OUT).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct TransformerModel<B: Backend> {
    // TODO: embedding layers
    attention_blocks: Vec<AttentionBlock<B>>,
    policy_head: PolicyHead<B>,
    value_head: ValueHead<B>,
}

#[derive(Config, Debug)]
pub(crate) struct TransformerModelConfig {
    #[config(default = 8)]
    n_blocks: usize,
    #[config(default = 6)]
    n_heads: usize,
    #[config(default = 2f64)]
    d_ff_scale: f64,
}

impl TransformerModelConfig {
    const HEAD_DIMENSION: usize = 64;

    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> TransformerModel<B> {
        let d_model = self.n_heads * Self::HEAD_DIMENSION;
        let d_ff = (self.d_ff_scale * d_model as f64) as usize;

        TransformerModel {
            attention_blocks: (0..self.n_blocks)
                .map(|_| {
                    AttentionBlockConfig::new()
                        .with_n_heads(self.n_heads)
                        .with_d_model(d_model)
                        .with_d_ff(d_ff)
                        .init(device)
                })
                .collect(),
            policy_head: PolicyHeadConfig::new()
                .with_d_model(d_model)
                .with_d_ff(d_model)
                .init(device),
            value_head: ValueHeadConfig::new()
                .with_d_model(d_model)
                .with_d_ff(d_model)
                .init(device),
        }
    }
}
