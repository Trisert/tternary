use burn::prelude::*;
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::activation::sigmoid;
use crate::ternary::TernaryLinear;
use rand::Rng;

#[derive(Module, Debug)]
pub struct TernaryEmbedding<B: Backend> {
    pub embedding: Embedding<B>,
}

impl<B: Backend> TernaryEmbedding<B> {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(num_embeddings, embedding_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(input)
    }
}

#[derive(Module, Debug)]
pub struct TernaryRMSNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> TernaryRMSNorm<B> {
    pub fn new(embed_dim: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([embed_dim], device)),
            eps: 1e-8,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let eps = self.eps;
        let dim = x.dims()[2] as f64;
        let rms = (x.clone().powf_scalar(2.0).sum_dim(2) / dim)
            .add_scalar(eps)
            .sqrt()
            .recip();
        let rms = rms.expand(x.dims());
        let [batch, seq, d] = x.dims();
        let w = self.weight.val().reshape([1, 1, d]).expand([batch, seq, d]);
        x * rms * w
    }
}

#[derive(Module, Debug)]
pub struct TernaryDepthwiseConv<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub kernel_size: usize,
    pub seq_len: usize,
}

impl<B: Backend> TernaryDepthwiseConv<B> {
    pub fn new(embed_dim: usize, kernel_size: usize, seq_len: usize, device: &B::Device) -> Self {
        let init_scale = (2.0 / (kernel_size as f32).sqrt()) * 0.5;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..embed_dim * kernel_size)
            .map(|_| rng.gen_range(-1.0f32..1.0) * init_scale)
            .collect();
        let weight = Param::from_tensor(
            Tensor::from_data(
                TensorData::new(data, [embed_dim, kernel_size]).convert::<B::FloatElem>(),
                device,
            ),
        );
        Self { weight, kernel_size, seq_len }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d] = x.dims();
        let ks = self.kernel_size;
        let x_padded = Tensor::zeros([batch, seq + ks - 1, d], &x.device());
        let x_padded = x_padded.slice_assign([0..batch, ks - 1..ks - 1 + seq, 0..d], x.clone());

        let weight = self.weight.val();
        let mut result = Tensor::zeros([batch, seq, d], &x.device());
        for k in 0..ks {
            let shifted = x_padded.clone().slice([0..batch, k..k + seq, 0..d]);
            let wt_col = weight.clone().slice([0..d, k..k + 1]).reshape([d]);
            let wt_bc = wt_col.reshape([1, 1, d]).expand([batch, seq, d]);
            result = result + shifted * wt_bc;
        }
        result
    }
}

#[derive(Module, Debug)]
pub struct GatedConvMixer<B: Backend> {
    pub conv: TernaryDepthwiseConv<B>,
    pub gate: TernaryLinear<B>,
}

impl<B: Backend> GatedConvMixer<B> {
    pub fn new(embed_dim: usize, kernel_size: usize, seq_len: usize, device: &B::Device) -> Self {
        Self {
            conv: TernaryDepthwiseConv::new(embed_dim, kernel_size, seq_len, device),
            gate: TernaryLinear::new(embed_dim, embed_dim, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d] = x.dims();
        let conv_out = self.conv.forward(x.clone());
        let x_2d = x.reshape([batch * seq, d]);
        let gate_raw = self.gate.forward(x_2d);
        let sig = sigmoid(gate_raw);
        let gate_3d = sig.reshape([batch, seq, d]);
        conv_out * gate_3d
    }
}

fn silu<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    x.clone() * sigmoid(x)
}

#[derive(Module, Debug)]
pub struct TernaryGLUFFN<B: Backend> {
    pub w_gate: TernaryLinear<B>,
    pub w_up: TernaryLinear<B>,
    pub w_down: TernaryLinear<B>,
}

impl<B: Backend> TernaryGLUFFN<B> {
    pub fn new(embed_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            w_gate: TernaryLinear::new(embed_dim, hidden_dim, device),
            w_up: TernaryLinear::new(embed_dim, hidden_dim, device),
            w_down: TernaryLinear::new(hidden_dim, embed_dim, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d] = x.dims();
        let x_2d = x.reshape([batch * seq, d]);
        let gate_raw = self.w_gate.forward(x_2d.clone());
        let up = self.w_up.forward(x_2d);
        let silu_gate = silu(gate_raw);
        let multiplied = silu_gate * up;
        let out = self.w_down.forward(multiplied);
        out.reshape([batch, seq, d])
    }
}

#[derive(Module, Debug)]
pub struct BoltBlock<B: Backend> {
    pub norm1: TernaryRMSNorm<B>,
    pub mixer: GatedConvMixer<B>,
    pub norm2: TernaryRMSNorm<B>,
    pub ffn: TernaryGLUFFN<B>,
}

impl<B: Backend> BoltBlock<B> {
    pub fn new(embed_dim: usize, hidden_dim: usize, kernel_size: usize, seq_len: usize, device: &B::Device) -> Self {
        Self {
            norm1: TernaryRMSNorm::new(embed_dim, device),
            mixer: GatedConvMixer::new(embed_dim, kernel_size, seq_len, device),
            norm2: TernaryRMSNorm::new(embed_dim, device),
            ffn: TernaryGLUFFN::new(embed_dim, hidden_dim, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed1 = self.norm1.forward(x.clone());
        let mixer_out = self.mixer.forward(normed1);
        let x2 = x + mixer_out;
        let normed2 = self.norm2.forward(x2.clone());
        let ffn_out = self.ffn.forward(normed2);
        x2 + ffn_out
    }
}
