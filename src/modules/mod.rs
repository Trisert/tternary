pub use crate::ternary::{TernaryWeight, TernaryLinear};
use ndarray::{Array1, Array2, Axis, s};

pub struct TernaryEmbedding {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub weight: TernaryWeight,
    last_input: Option<Array2<f32>>,
}

impl TernaryEmbedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self { num_embeddings, embedding_dim, weight: TernaryWeight::new((num_embeddings, embedding_dim)), last_input: None }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        let rows = input.shape()[0];
        let mut result = Array2::zeros((rows, self.embedding_dim));
        let fw = &self.weight.float_weights;
        for i in 0..rows {
            let token = input[[i, 0]] as usize;
            if token < self.num_embeddings {
                result.row_mut(i).assign(&fw.row(token));
            }
        }
        result
    }

    pub fn backward(&mut self, grad: &Array2<f32>, lr: f32) {
        let input = match self.last_input.take() { Some(x) => x, None => return };
        let rows = grad.shape()[0].min(input.shape()[0]);
        let (n_emb, emb_dim) = (self.num_embeddings, self.embedding_dim);
        let mut grad_w: Array2<f32> = Array2::zeros((n_emb, emb_dim));
        for i in 0..rows {
            let token = input[[i, 0]] as usize;
            if token < n_emb {
                let limit = emb_dim.min(grad.shape()[1]);
                let mut gw_row = grad_w.row_mut(token);
                let g_row = grad.row(i);
                for j in 0..limit {
                    gw_row[j] += g_row[j];
                }
            }
        }
        self.weight.update(&grad_w, lr);
    }

    pub fn requantize(&mut self) { self.weight.quantize(); }
}

pub struct TernaryRMSNorm {
    weight: Array2<f32>,
    eps: f32,
}

impl TernaryRMSNorm {
    pub fn new(embed_dim: usize) -> Self {
        Self { weight: Array2::ones((1, embed_dim)), eps: 1e-8 }
    }

    pub fn forward_data(&self, x: &Array2<f32>) -> Array2<f32> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let rms_sq = (x * x).sum_axis(Axis(1)) / cols as f32;
        let inv_rms = rms_sq.mapv(|v| 1.0 / (v + self.eps).sqrt());
        let w = self.weight.row(0);
        let mut result = x.clone();
        for i in 0..rows {
            let ir = inv_rms[i];
            result.row_mut(i).zip_mut_with(&w, |r, &ww| *r *= ir * ww);
        }
        result
    }

    pub fn backward_data(&mut self, grad_output: &Array2<f32>, x: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let rms_sq = (x * x).sum_axis(Axis(1)) / cols as f32;
        let inv_rms = rms_sq.mapv(|v| 1.0 / (v + self.eps).sqrt());

        let w = self.weight.row(0);
        let mut grad_input = grad_output.clone();
        for i in 0..rows {
            let ir = inv_rms[i];
            grad_input.row_mut(i).zip_mut_with(&w, |gi, &ww| *gi *= ww * ir);
        }

        let mut grad_w_accum = Array1::zeros(cols);
        for i in 0..rows {
            let ir = inv_rms[i];
            let x_row = x.row(i);
            let go_row = grad_output.row(i);
            let mut gw = grad_w_accum.view_mut();
            for j in 0..cols {
                gw[j] += x_row[j] * ir * go_row[j];
            }
        }

        let update = grad_w_accum.mapv(|g: f32| -> f32 { (g / rows as f32).max(-0.1).min(0.1) * lr });
        self.weight.row_mut(0).zip_mut_with(&update, |w, &u| *w -= u);

        grad_input
    }

    pub fn requantize(&mut self) {}
}

pub struct TernaryDepthwiseConv {
    pub weight: TernaryWeight,
    seq_len: usize,
    kernel_size: usize,
}

impl TernaryDepthwiseConv {
    pub fn new(embed_dim: usize, kernel_size: usize, seq_len: usize) -> Self {
        Self { weight: TernaryWeight::new((embed_dim, kernel_size)), seq_len, kernel_size }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let total = x.shape()[0];
        let d = x.shape()[1];
        let batch = total / self.seq_len;
        let ks = self.kernel_size;
        let seq_len = self.seq_len;
        let wt = &self.weight.float_weights;

        let mut result = Array2::zeros((total, d));

        for b in 0..batch {
            let base = b * seq_len;
            let mut padded = Array2::zeros((seq_len + ks - 1, d));
            padded.slice_mut(s![ks - 1.., ..]).assign(&x.slice(s![base..base + seq_len, ..]));

            let mut batch_result = Array2::zeros((seq_len, d));
            for k in 0..ks {
                let shifted = padded.slice(s![k..k + seq_len, ..]);
                let wt_col_2d = wt.column(k).to_owned().insert_axis(Axis(0));
                let bc = wt_col_2d.broadcast((seq_len, d)).unwrap();
                batch_result += &(&shifted * &bc);
            }

            result.slice_mut(s![base..base + seq_len, ..]).assign(&batch_result);
        }

        result
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, x: &Array2<f32>, lr: f32) -> Array2<f32> {
        let total = x.shape()[0];
        let d = x.shape()[1];
        let batch = total / self.seq_len;
        let ks = self.kernel_size;
        let seq_len = self.seq_len;
        let wt = &self.weight.float_weights;

        let mut grad_input = Array2::zeros((total, d));
        let mut grad_w = Array2::zeros((d, ks));

        for b in 0..batch {
            let base = b * seq_len;
            let x_batch = x.slice(s![base..base + seq_len, ..]);
            let go_batch = grad_output.slice(s![base..base + seq_len, ..]);

            let mut padded_x = Array2::zeros((seq_len + ks - 1, d));
            padded_x.slice_mut(s![ks - 1.., ..]).assign(&x_batch);

            let mut padded_go = Array2::zeros((seq_len + ks - 1, d));
            padded_go.slice_mut(s![..seq_len, ..]).assign(&go_batch);

            let mut gi_batch = Array2::zeros((seq_len, d));

            for k in 0..ks {
                let shifted_x = padded_x.slice(s![k..k + seq_len, ..]);
                let prod = &go_batch * &shifted_x;
                let col_sum = prod.sum_axis(Axis(0));
                grad_w.column_mut(k).zip_mut_with(&col_sum, |gw, &cs| *gw += cs);

                let shift = ks - 1 - k;
                let shifted_go = padded_go.slice(s![shift..shift + seq_len, ..]);
                let wt_col_2d = wt.column(k).to_owned().insert_axis(Axis(0));
                let bc = wt_col_2d.broadcast((seq_len, d)).unwrap();
                gi_batch += &(&shifted_go * &bc);
            }

            grad_input.slice_mut(s![base..base + seq_len, ..]).assign(&gi_batch);
        }

        self.weight.update(&grad_w, lr);
        grad_input
    }

    pub fn requantize(&mut self) { self.weight.quantize(); }
}

pub struct GatedConvMixer {
    conv: TernaryDepthwiseConv,
    gate: TernaryLinear,
    stored_conv_out: Option<Array2<f32>>,
    stored_sig: Option<Array2<f32>>,
}

impl GatedConvMixer {
    pub fn new(embed_dim: usize, kernel_size: usize, seq_len: usize) -> Self {
        Self {
            conv: TernaryDepthwiseConv::new(embed_dim, kernel_size, seq_len),
            gate: TernaryLinear::new((embed_dim, embed_dim)),
            stored_conv_out: None,
            stored_sig: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let conv_out = self.conv.forward(x);
        let gate_raw = self.gate.forward(x);
        let sig = gate_raw.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let result = &conv_out * &sig;
        self.stored_conv_out = Some(conv_out);
        self.stored_sig = Some(sig);
        result
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, x: &Array2<f32>, lr: f32) -> Array2<f32> {
        let conv_out = self.stored_conv_out.take().unwrap();
        let sig = self.stored_sig.take().unwrap();

        let grad_conv = grad_output * &sig;
        let one_minus_sig = sig.mapv(|s| 1.0 - s);
        let grad_gate_raw = grad_output * &conv_out * &sig * &one_minus_sig;

        self.gate.last_input = Some(x.clone());
        let grad_gate_input = self.gate.backward(&grad_gate_raw, lr);
        let grad_conv_input = self.conv.backward(&grad_conv, x, lr);
        grad_conv_input + grad_gate_input
    }

    pub fn requantize(&mut self) {
        self.conv.requantize();
        self.gate.requantize();
    }
}

fn silu(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    x * sig
}

fn silu_deriv(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig + x * sig * (1.0 - sig)
}

pub struct TernaryGLUFFN {
    w_gate: TernaryLinear,
    w_up: TernaryLinear,
    w_down: TernaryLinear,
    stored_gate_raw: Option<Array2<f32>>,
    stored_up: Option<Array2<f32>>,
}

impl TernaryGLUFFN {
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        Self {
            w_gate: TernaryLinear::new((embed_dim, hidden_dim)),
            w_up: TernaryLinear::new((embed_dim, hidden_dim)),
            w_down: TernaryLinear::new((hidden_dim, embed_dim)),
            stored_gate_raw: None,
            stored_up: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let gate_raw = self.w_gate.forward(x);
        let up = self.w_up.forward(x);
        let silu_gate = gate_raw.mapv(silu);
        let multiplied = &silu_gate * &up;
        self.stored_gate_raw = Some(gate_raw);
        self.stored_up = Some(up);
        self.w_down.forward(&multiplied)
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let gate_raw = self.stored_gate_raw.take().unwrap();
        let up = self.stored_up.take().unwrap();

        let grad_multiplied = self.w_down.backward(grad_output, lr);

        let silu_gate = gate_raw.mapv(silu);
        let silu_deriv_gate = gate_raw.mapv(silu_deriv);
        let grad_gate_raw = &grad_multiplied * &up * &silu_deriv_gate;
        let grad_up = &grad_multiplied * &silu_gate;

        let g1 = self.w_gate.backward(&grad_gate_raw, lr);
        let g2 = self.w_up.backward(&grad_up, lr);
        g1 + g2
    }

    pub fn requantize(&mut self) {
        self.w_gate.requantize();
        self.w_up.requantize();
        self.w_down.requantize();
    }
}

pub struct BoltBlock {
    norm1: TernaryRMSNorm,
    mixer: GatedConvMixer,
    norm2: TernaryRMSNorm,
    ffn: TernaryGLUFFN,
    last_x1: Option<Array2<f32>>,
    last_x2: Option<Array2<f32>>,
    last_normed1: Option<Array2<f32>>,
}

impl BoltBlock {
    pub fn new(embed_dim: usize, hidden_dim: usize, kernel_size: usize, seq_len: usize) -> Self {
        Self {
            norm1: TernaryRMSNorm::new(embed_dim),
            mixer: GatedConvMixer::new(embed_dim, kernel_size, seq_len),
            norm2: TernaryRMSNorm::new(embed_dim),
            ffn: TernaryGLUFFN::new(embed_dim, hidden_dim),
            last_x1: None,
            last_x2: None,
            last_normed1: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.last_x1 = Some(x.clone());
        let normed1 = self.norm1.forward_data(x);
        let mixer_out = self.mixer.forward(&normed1);
        self.last_normed1 = Some(normed1);
        let x2 = x + mixer_out;
        self.last_x2 = Some(x2.clone());
        let normed2 = self.norm2.forward_data(&x2);
        let ffn_out = self.ffn.forward(&normed2);
        x2 + ffn_out
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let x2 = self.last_x2.take().unwrap();
        let x = self.last_x1.take().unwrap();
        let normed1 = self.last_normed1.take().unwrap();

        let grad_normed2 = self.ffn.backward(grad_output, lr);
        let grad_x2_from_ffn = self.norm2.backward_data(&grad_normed2, &x2, lr);
        let grad_x2 = grad_x2_from_ffn + grad_output;

        let grad_normed1 = self.mixer.backward(&grad_x2, &normed1, lr);
        let grad_x_from_mixer = self.norm1.backward_data(&grad_normed1, &x, lr);

        grad_x_from_mixer + grad_x2
    }

    pub fn requantize(&mut self) {
        self.mixer.requantize();
        self.ffn.requantize();
    }
}
