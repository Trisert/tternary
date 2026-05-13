use burn::prelude::*;
use burn::module::Param;
use burn::tensor::TensorData;
use rand::Rng;

#[derive(Module, Debug)]
pub struct TernaryLinear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> TernaryLinear<B> {
    pub fn new(rows: usize, cols: usize, device: &B::Device) -> Self {
        let init_scale = (2.0 / (rows as f32).sqrt()) * 0.5;
        let mut rng = rand::thread_rng();
        let w_data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0f32..1.0) * init_scale)
            .collect();
        let weight = Param::from_tensor(
            Tensor::from_data(TensorData::new(w_data, [rows, cols]).convert::<B::FloatElem>(), device),
        );
        let bias = Param::from_tensor(Tensor::zeros([cols], device));
        Self { weight, bias }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = input.matmul(self.weight.val());
        let [rows, cols] = out.dims();
        out + self.bias.val().reshape([1, cols]).expand([rows, cols])
    }
}
