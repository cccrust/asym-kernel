use anyhow::Result;
use candle_core::{Tensor, DType};

#[derive(Debug, Clone)]
pub enum Operator {
    Square,
    Softmax,
    AddScalar(f32),
    Normalize,
    MatrixSort, // 新增：矩陣排序算子
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
// --- AI 專屬：矩陣排序 ---
            Operator::MatrixSort => {
                // 1. 使用 true 進行遞增排序 (Ascending)
                let sorted_indices = tensor.arg_sort_last_dim(true)?;
                
                // 2. 使用 gather 根據索引重排數據
                let out = tensor.gather(&sorted_indices, 0)?;
                
                Ok(out)
            }
        }
    }
}
