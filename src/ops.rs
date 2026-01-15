use anyhow::Result;
use candle_core::Tensor;

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
    // 新增：儲存當前張量到指定的 Key
    Store(String),
    // 新增：從指定的 Key 加載張量並替換當前張量
    Load(String),
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
            // Branch, Store, Load 由 Kernel 執行流程控制，apply 階段直接回傳
            _ => Ok(tensor),
        }
    }
}