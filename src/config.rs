use burn::config::Config;

#[derive(Config, Debug)]
pub struct AppConfig {
    pub vocab_size: usize,
    #[config(default = 256)]
    pub embed_dim: usize,
    #[config(default = 512)]
    pub hidden_dim: usize,
    #[config(default = 6)]
    pub num_layers: usize,
    #[config(default = 512)]
    pub max_seq_len: usize,
    #[config(default = 16)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub kernel_size: usize,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 500)]
    pub steps_per_epoch: usize,
    #[config(default = 0.003)]
    pub learning_rate: f64,
}
