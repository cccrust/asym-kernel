use anyhow::Result;
use candle_core::Tensor; // 移除 DType

#[derive(Debug, Clone)]
pub enum Operator {
    Square,
    Softmax,
    AddScalar(f32),
    Normalize,
    MatrixSort,
    Branch {
        threshold: f32,
        true_path: Vec<Operator>,
        false_path: Vec<Operator>,
    },
}

impl Operator {
    pub fn apply(&self, tensor: Tensor) -> Result<Tensor> {
        match self {
            Operator::Square => Ok(tensor.sqr()?),
            Operator::Softmax => Ok(candle_nn::ops::softmax(&tensor, 0)?),
            Operator::AddScalar(val) => Ok(tensor.affine(1.0, *val as f64)?),
            Operator::Normalize => {
                let mean = tensor.mean_all()?;
                let centered = tensor.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_all()?;
                let std = var.affine(1.0, 1e-5)?.sqrt()?;
                Ok(centered.broadcast_div(&std)?)
            }
            Operator::MatrixSort => {
                let sorted_indices = tensor.arg_sort_last_dim(true)?;
                Ok(tensor.gather(&sorted_indices, 0)?)
            }
            // Branch 由 Kernel 的 run_step 處理，這裡直接回傳
            Operator::Branch { .. } => Ok(tensor),
        }
    }
}