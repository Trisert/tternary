use burn::prelude::*;
use burn::module::Param;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, InferenceStep};
use crate::config::AppConfig;
use crate::modules::{TernaryEmbedding, BoltBlock};
use rand::Rng;

#[derive(Module, Debug)]
pub struct TernaryTransformer<B: Backend> {
    token_embed: TernaryEmbedding<B>,
    pos_embed: TernaryEmbedding<B>,
    blocks: Vec<BoltBlock<B>>,
    norm: crate::modules::TernaryRMSNorm<B>,
    output_weight: Param<Tensor<B, 2>>,
    vocab_size: usize,
    max_seq_len: usize,
}

impl AppConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TernaryTransformer<B> {
        let token_embed = TernaryEmbedding::new(self.vocab_size, self.embed_dim, device);
        let pos_embed = TernaryEmbedding::new(self.max_seq_len, self.embed_dim, device);
        let mut blocks = Vec::new();
        for _ in 0..self.num_layers {
            blocks.push(BoltBlock::new(
                self.embed_dim,
                self.hidden_dim,
                self.kernel_size,
                self.max_seq_len,
                device,
            ));
        }
        let norm = crate::modules::TernaryRMSNorm::new(self.embed_dim, device);

        let mut rng = rand::thread_rng();
        let init_scale = (2.0_f32 / self.embed_dim as f32).sqrt();
        let data: Vec<f32> = (0..self.embed_dim * self.vocab_size)
            .map(|_| rng.gen_range(-1.0f32..1.0) * init_scale)
            .collect();
        let output_weight = Param::from_tensor(
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [self.embed_dim, self.vocab_size]).convert::<B::FloatElem>(),
                device,
            ),
        );

        TernaryTransformer {
            token_embed,
            pos_embed,
            blocks,
            norm,
            output_weight,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
        }
    }
}

impl<B: Backend> TernaryTransformer<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = input_ids.dims();
        let device = &input_ids.device();

        let positions = Tensor::arange(0..seq as i64, device)
            .reshape([1, seq])
            .repeat_dim(0, batch);

        let token_emb = self.token_embed.forward(input_ids);
        let pos_emb = self.pos_embed.forward(positions);
        let mut x = token_emb + pos_emb;

        for block in &self.blocks {
            x = block.forward(x);
        }

        self.norm.forward(x)
    }

    pub fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let hidden = self.forward(input_ids);
        let [batch, seq, d] = hidden.dims();
        let hidden_2d = hidden.reshape([batch * seq, d]);
        let logits = hidden_2d.matmul(self.output_weight.val());
        logits.reshape([batch, seq, self.vocab_size])
    }

    pub fn forward_training(&self, input_ids: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> ClassificationOutput<B> {
        let [batch, seq] = input_ids.dims();
        let logits = self.forward_logits(input_ids);
        let logits_flat = logits.reshape([batch * seq, self.vocab_size]);
        let targets_flat = targets.reshape([batch * seq]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device());
        let loss = loss.forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: logits_flat,
            targets: targets_flat,
        }
    }

    pub fn num_parameters(&self) -> usize {
        let mut total = 0usize;
        total += self.token_embed.embedding.weight.val().shape().num_elements();
        total += self.pos_embed.embedding.weight.val().shape().num_elements();
        total += self.output_weight.val().shape().num_elements();
        total += self.norm.weight.val().shape().num_elements();
        for block in &self.blocks {
            total += block.norm1.weight.val().shape().num_elements();
            total += block.norm2.weight.val().shape().num_elements();
            total += block.mixer.conv.conv1d.weight.val().shape().num_elements();
            total += block.mixer.gate.weight.val().shape().num_elements();
            total += block.mixer.gate.bias.val().shape().num_elements();
            total += block.ffn.w_gate.weight.val().shape().num_elements();
            total += block.ffn.w_gate.bias.val().shape().num_elements();
            total += block.ffn.w_up.weight.val().shape().num_elements();
            total += block.ffn.w_up.bias.val().shape().num_elements();
            total += block.ffn.w_down.weight.val().shape().num_elements();
            total += block.ffn.w_down.bias.val().shape().num_elements();
        }
        total
    }
}

#[derive(Clone, Debug)]
pub struct TernaryTransformerTrainingBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: AutodiffBackend> TrainStep for TernaryTransformer<B> {
    type Input = TernaryTransformerTrainingBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TernaryTransformerTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward_training(batch.inputs, batch.targets);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for TernaryTransformer<B> {
    type Input = TernaryTransformerTrainingBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TernaryTransformerTrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(batch.inputs, batch.targets)
    }
}
