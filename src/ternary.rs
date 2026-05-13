use burn::prelude::*;
use burn::tensor::TensorData;
use rand::Rng;

pub struct TernaryWeight<B: Backend> {
    pub float_weights: Tensor<B, 2>,
    pub scale: f32,
}

impl<B: Backend> TernaryWeight<B> {
    pub fn new(rows: usize, cols: usize, device: &B::Device) -> Self {
        let init_scale = (2.0 / (rows as f32).sqrt()) * 0.5;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0f32..1.0) * init_scale)
            .collect();
        let float_weights =
            Tensor::from_data(TensorData::new(data, [rows, cols]).convert::<B::FloatElem>(), device);

        let mut tw = Self {
            float_weights,
            scale: 1.0,
        };
        tw.quantize();
        tw
    }

    pub fn quantize(&mut self) {
        let weights = self.float_weights.clone().to_data().to_vec::<f32>().unwrap();
        let abs_mean = weights.iter().map(|v| v.abs()).sum::<f32>() / weights.len() as f32;
        self.scale = abs_mean.max(1e-8);
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        input.matmul(self.float_weights.clone().transpose())
    }

    pub fn shape(&self) -> [usize; 2] {
        self.float_weights.dims()
    }

    pub fn dead_fraction(&self) -> f32 {
        let weights = self.float_weights.clone().to_data().to_vec::<f32>().unwrap();
        let zeros = weights.iter().filter(|&&v| v.abs() <= self.scale).count();
        zeros as f32 / weights.len() as f32
    }
}

#[derive(Module, Debug)]
pub struct TernaryLinear<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Tensor<B, 1>,
}

impl<B: Backend> TernaryLinear<B> {
    pub fn new(rows: usize, cols: usize, device: &B::Device) -> Self {
        let init_scale = (2.0 / (rows as f32).sqrt()) * 0.5;
        let mut rng = rand::thread_rng();
        let w_data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0f32..1.0) * init_scale)
            .collect();
        let weight =
            Tensor::from_data(TensorData::new(w_data, [rows, cols]).convert::<B::FloatElem>(), device);
        let bias = Tensor::zeros([cols], device);
        Self { weight, bias }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = input.matmul(self.weight.clone());
        let [rows, cols] = out.dims();
        out + self.bias.clone().reshape([1, cols]).expand([rows, cols])
    }
}
