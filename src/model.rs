use crate::config::Config;
use crate::modules::{TernaryEmbedding, BoltBlock, TernaryRMSNorm};
use ndarray::{Array2, s};

pub struct TernaryTransformer {
    token_embed: TernaryEmbedding,
    pos_embed: TernaryEmbedding,
    blocks: Vec<BoltBlock>,
    norm: TernaryRMSNorm,
    config: Config,
    vocab_size: usize,
}

impl TernaryTransformer {
    pub fn new(config: Config) -> Self {
        let token_embed = TernaryEmbedding::new(config.vocab_size, config.embed_dim);
        let pos_embed = TernaryEmbedding::new(config.max_seq_len, config.embed_dim);
        let mut blocks = Vec::new();
        for _ in 0..config.num_layers {
            blocks.push(BoltBlock::new(config.embed_dim, config.hidden_dim, config.kernel_size, config.max_seq_len));
        }
        let norm = TernaryRMSNorm::new(config.embed_dim);
        Self { token_embed, pos_embed, blocks, norm, config, vocab_size: config.vocab_size }
    }

    pub fn forward(&mut self, input_ids: &Array2<f32>) -> Array2<f32> {
        let (batch_size, seq_len) = (input_ids.shape()[0], input_ids.shape()[1]);
        let total_tokens = batch_size * seq_len;

        let flat_input = input_ids.clone().into_shape((total_tokens, 1)).unwrap();
        let positions = Array2::from_shape_fn((total_tokens, 1), |(i, _)| (i % seq_len) as f32);

        let token_emb = self.token_embed.forward(&flat_input);
        let pos_emb = self.pos_embed.forward(&positions);
        let mut x = token_emb + pos_emb;

        for block in self.blocks.iter_mut() {
            x = block.forward(&x);
        }

        let x = self.norm.forward_data(&x);
        x.dot(&self.token_embed.weight.float_weights.t())
    }

    pub fn train_step(&mut self, input_arr: &Array2<f32>, target_arr: &Array2<f32>, lr: f32) -> f32 {
        let (batch_size, seq_len) = (input_arr.shape()[0], input_arr.shape()[1]);
        let total_tokens = batch_size * seq_len;

        let flat_input = input_arr.clone().into_shape((total_tokens, 1)).unwrap();
        let positions = Array2::from_shape_fn((total_tokens, 1), |(i, _)| (i % seq_len) as f32);

        let token_emb = self.token_embed.forward(&flat_input);
        let pos_emb = self.pos_embed.forward(&positions);
        let mut x = token_emb + pos_emb;

        for block in self.blocks.iter_mut() {
            x = block.forward(&x);
        }

        let x_pre_norm = x.clone();
        let final_hidden = self.norm.forward_data(&x);
        let logits = final_hidden.dot(&self.token_embed.weight.float_weights.t());

        let (rows, cols) = (logits.shape()[0], logits.shape()[1]);
        let vocab = self.vocab_size.min(cols);

        let mut total_loss = 0.0_f32;
        let mut count = 0;
        let mut grad_logits = Array2::zeros((rows, cols));

        for i in 0..rows {
            let target_row = i / seq_len;
            let target_col = i % seq_len;
            if target_row >= target_arr.shape()[0] || target_col >= target_arr.shape()[1] { continue; }
            let target_idx = target_arr[[target_row, target_col]] as usize;
            if target_idx >= vocab { continue; }

            let logit_row = logits.row(i);
            let max_val = logit_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let shifted = logit_row.mapv(|v| (v - max_val).exp());
            let sum_exp: f32 = shifted.sum();

            total_loss -= logit_row[target_idx] - max_val - sum_exp.ln();
            count += 1;

            let inv_sum = 1.0 / sum_exp;
            let mut grad_row = grad_logits.row_mut(i);
            grad_row.slice_mut(s![..vocab]).zip_mut_with(&shifted.slice(s![..vocab]), |g, &e| *g = e * inv_sum);
            grad_row[target_idx] -= 1.0;
        }

        let loss = if count > 0 { total_loss / count as f32 } else { 0.0 };
        if count > 0 {
            let scale = 1.0 / count as f32;
            grad_logits.mapv_inplace(|v| v * scale);
        }

        let grad_norm = grad_logits.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        if grad_norm > 1.0 {
            let clip = 1.0 / grad_norm;
            grad_logits.mapv_inplace(|v| v * clip);
        }

        let grad_final_hidden = grad_logits.dot(&self.token_embed.weight.float_weights);
        let grad_emb_w = grad_logits.t().dot(&final_hidden);
        self.token_embed.weight.update(&grad_emb_w, lr);

        let grad_x = self.norm.backward_data(&grad_final_hidden, &x_pre_norm, lr);

        let mut grad_through = grad_x;
        for block in self.blocks.iter_mut().rev() {
            grad_through = block.backward(&grad_through, lr);
        }

        self.token_embed.backward(&grad_through, lr);
        self.pos_embed.backward(&grad_through, lr);

        loss
    }

    pub fn num_parameters(&self) -> usize {
        let c = &self.config;
        let embed_params = c.vocab_size * c.embed_dim + c.max_seq_len * c.embed_dim;
        let per_block = c.embed_dim * c.kernel_size
            + c.embed_dim * c.embed_dim
            + c.embed_dim * c.hidden_dim * 2
            + c.hidden_dim * c.embed_dim
            + c.embed_dim * 2;
        embed_params + c.num_layers * per_block + c.embed_dim
    }

    pub fn requantize(&mut self) {
        self.token_embed.requantize();
        self.pos_embed.requantize();
        for block in &mut self.blocks {
            block.requantize();
        }
        self.norm.requantize();
    }
}
