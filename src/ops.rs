use anyhow::Result;
use candle_core::Tensor;

#[derive(Debug, Clone)]
pub enum Operator {
    Square,
    Softmax,
    AddScalar(f32),
    Normalize,
    MatrixSort, // 確保它在這裡
    Branch {
        threshold: f32,
        true_path: Vec<Operator>,
        false_path: Vec<Operator>,
    },
    Store(String),
    Load(String),
    AddParam(String),
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
            // 這裡最重要！必須明確處理 MatrixSort
            Operator::MatrixSort => {
                let sorted_indices = tensor.arg_sort_last_dim(true)?;
                // 在 1D 張量中使用 index_select 是最穩定的做法
                Ok(tensor.index_select(&sorted_indices, 0)?)
            }
            // 其他由 Kernel 處理的算子回傳原張量
            _ => Ok(tensor),
        }
    }
}