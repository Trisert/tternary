use burn::prelude::*;
use burn::module::Param;
use burn::tensor::{ElementConversion, TensorData};
use rand::Rng;

#[derive(Module, Debug)]
pub struct TernaryLinear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> TernaryLinear<B> {
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let init_scale = (2.0_f32 / in_dim as f32).sqrt();
        let mut rng = rand::thread_rng();
        let w_data: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| rng.gen_range(-1.0f32..1.0) * init_scale)
            .collect();
        let weight = Param::from_tensor(
            Tensor::from_data(
                TensorData::new(w_data, [in_dim, out_dim]).convert::<B::FloatElem>(),
                device,
            ),
        );
        let bias = Param::from_tensor(Tensor::zeros([out_dim], device));
        Self { weight, bias }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let w = self.weight.val();
        let w_detached = w.clone().detach();

        let scale_val: f32 = w_detached.clone().abs().mean().into_scalar().elem::<f32>();
        let threshold = scale_val * 0.5;

        let pos = w_detached.clone().greater_elem(threshold).float();
        let neg = w_detached.clone().lower_elem(-threshold).float();
        let w_ternary = (pos - neg) * scale_val;

        let w_ste = w_ternary + (w - w_detached);

        let out = input.matmul(w_ste);
        let [rows, cols] = out.dims();
        out + self.bias.val().reshape([1, cols]).expand([rows, cols])
    }
}
