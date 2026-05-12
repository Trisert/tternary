#[derive(Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub max_grad_norm: f32,
    pub kernel_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 3000,
            embed_dim: 256,
            hidden_dim: 512,
            num_layers: 6,
            max_seq_len: 512,
            batch_size: 16,
            learning_rate: 0.003,
            max_grad_norm: 1.0,
            kernel_size: 8,
        }
    }
}
