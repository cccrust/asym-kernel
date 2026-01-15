use anyhow::Result;
use candle_core::Tensor;

#[derive(Debug, Clone)]
pub enum Operator {
    Square,           // 平方
    Softmax,          // 歸一化
    AddScalar(f32),   // 加法
    Normalize,        // 正規化 (Mean=0, Std=1)
}

impl Operator {
    pub fn apply(&self, tensor: Tensor) -> Result<Tensor> {
        match self {
            Operator::Square => Ok(tensor.sqr()?),
            
            // Softmax: 對 1D 張量在維度 0 進行歸一化
            Operator::Softmax => Ok(candle_nn::ops::softmax(&tensor, 0)?),
            
            // 使用 affine(1.0, val) 實現加法: y = 1.0 * x + val
            Operator::AddScalar(val) => Ok(tensor.affine(1.0, *val as f64)?),
            
            // Normalize: (x - mean) / sqrt(var + eps)
            Operator::Normalize => {
                let mean = tensor.mean_all()?;
                // x - mean
                let centered = tensor.broadcast_sub(&mean)?;
                
                // Variance = Mean((x - mean)^2)
                let var = centered.sqr()?.mean_all()?;
                
                // Std = sqrt(var + 1e-5)
                // 這裡同樣使用 affine 加上 epsilon
                let std = var.affine(1.0, 1e-5)?.sqrt()?;
                
                // (x - mean) / std
                Ok(centered.broadcast_div(&std)?)
            }
        }
    }
}