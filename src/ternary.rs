use ndarray::{Array2, Axis};
use rand::Rng;

pub struct TernaryWeight {
    pub ternary: Array2<i8>,
    pub float_weights: Array2<f32>,
    pub scale: f32,
    dead_steps: Array2<u32>,
}

impl TernaryWeight {
    pub fn new(shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;
        let mut rng = rand::thread_rng();
        let init_scale = (2.0 / (rows as f32).sqrt()) * 0.5;
        let float_weights = Array2::from_shape_fn((rows, cols), |_| {
            rng.gen_range(-1.0f32..1.0) * init_scale
        });
        let mut tw = Self {
            ternary: Array2::zeros((rows, cols)),
            float_weights,
            scale: 1.0,
            dead_steps: Array2::zeros((rows, cols)),
        };
        tw.quantize();
        tw
    }

    pub fn quantize(&mut self) {
        let scale = self.float_weights.mapv(|v| v.abs()).mean().unwrap_or(1.0).max(1e-8);
        self.scale = scale;
        let s = scale;
        self.ternary = self.float_weights.mapv(move |w| {
            if w > s { 1 } else if w < -s { -1 } else { 0 }
        });
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.float_weights)
    }

    pub fn update(&mut self, grad: &Array2<f32>, lr: f32) {
        let mut rng = rand::thread_rng();
        let dead_threshold = 20u32;
        let (rows, cols) = (self.float_weights.shape()[0], self.float_weights.shape()[1]);

        let clipped = grad.mapv(|g| g.max(-1.0f32).min(1.0f32));
        {
            let lr_scaled = &clipped * lr;
            self.float_weights = &self.float_weights - &lr_scaled;
        }

        for i in 0..rows {
            for j in 0..cols {
                let w = self.float_weights[[i, j]];
                if w > self.scale {
                    self.ternary[[i, j]] = 1;
                    self.dead_steps[[i, j]] = 0;
                } else if w < -self.scale {
                    self.ternary[[i, j]] = -1;
                    self.dead_steps[[i, j]] = 0;
                } else {
                    self.ternary[[i, j]] = 0;
                    self.dead_steps[[i, j]] += 1;
                    if self.dead_steps[[i, j]] >= dead_threshold {
                        let sign = if rng.gen_range(0..2) == 0 { 1.0f32 } else { -1.0f32 };
                        self.float_weights[[i, j]] = sign * self.scale * (1.0 + rng.gen_range(0.0f32..0.5));
                        self.dead_steps[[i, j]] = 0;
                    }
                }
            }
        }

        self.float_weights.mapv_inplace(|v| v.max(-5.0).min(5.0));
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.ternary.shape()[0], self.ternary.shape()[1])
    }

    pub fn as_float(&self) -> Array2<f32> {
        self.float_weights.clone()
    }

    pub fn dead_fraction(&self) -> f32 {
        let total = self.ternary.len() as f32;
        let zeros = self.ternary.iter().filter(|&&v| v == 0).count() as f32;
        zeros / total
    }
}

pub struct TernaryLinear {
    pub weight: TernaryWeight,
    pub bias: Array2<f32>,
    pub last_input: Option<Array2<f32>>,
}

impl TernaryLinear {
    pub fn new(shape: (usize, usize)) -> Self {
        Self { weight: TernaryWeight::new(shape), bias: Array2::zeros((1, shape.1)), last_input: None }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        let mut result = self.weight.forward(input);
        let b = self.bias.row(0);
        for mut row in result.rows_mut() {
            row += &b;
        }
        result
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = match self.last_input.take() {
            Some(x) => x,
            None => return Array2::zeros((grad_output.shape()[0], self.weight.shape().0)),
        };

        let grad_w = input.t().dot(grad_output);
        self.weight.update(&grad_w, lr);

        let batch = grad_output.shape()[0];
        let inv_batch = 1.0 / batch as f32;
        let grad_bias = grad_output.sum_axis(Axis(0));
        let clipped_grad_bias = grad_bias.mapv(|g| (g * inv_batch).max(-1.0f32).min(1.0f32));
        let update = &clipped_grad_bias * lr;
        self.bias.row_mut(0).zip_mut_with(&update, |b, &u| *b -= u);

        grad_output.dot(&self.weight.float_weights.t())
    }

    pub fn requantize(&mut self) {
        self.weight.quantize();
    }
}
