use candle_core::{Device, Tensor};

pub struct HyperTensor {
    pub data: Tensor,
    pub label: String, // 用於語義追蹤
}

impl HyperTensor {
    pub fn new(shape: &[usize], device: &Device) -> Self {
        let data = Tensor::zeros(shape, candle_core::DType::F32, device).unwrap();
        Self { data, label: "raw_tensor".to_string() }
    }
}