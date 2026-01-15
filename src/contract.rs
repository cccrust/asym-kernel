// src/contract.rs
use crate::tensor::HyperTensor;
use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub enum Invariant {
    IsMonotonic,
    MeanEquals(f32),
    AlwaysTrue,
}

#[derive(Clone)]
pub struct Contract {
    pub expected_shape: Vec<usize>,
    pub invariant: Invariant,
}

impl Contract {
    pub fn verify(&self, tensor: &HyperTensor) -> Result<()> {
        // 1. 無論輸入輸出，形狀 (Shape) 必須始終符合
        if tensor.data.dims() != self.expected_shape {
            bail!("Shape Mismatch: expected {:?}, found {:?}", self.expected_shape, tensor.data.dims());
        }

        // 2. 關鍵修正：如果是輸入數據 (包含 "input" 字樣)，跳過後續的數學不變量檢查
        // 這樣才能讓亂序數據進入排序算子
        if tensor.label.contains("input") {
            return Ok(());
        }

        // 3. 只有對輸出數據才執行數學驗證
        match self.invariant {
            Invariant::IsMonotonic => {
                if let Ok(vals) = tensor.data.to_vec1::<f32>() {
                    if !vals.windows(2).all(|w| w[0] <= w[1]) {
                        bail!("Invariant Violation: Data is not monotonic (Sorted result failed!)");
                    }
                }
            }
            Invariant::MeanEquals(target) => {
                let mean = tensor.data.mean_all()?.to_scalar::<f32>()?;
                if (mean - target).abs() > 0.001 {
                    bail!("Invariant Violation: Mean is {:.4}, expected {}", mean, target);
                }
            }
            Invariant::AlwaysTrue => {}
        }
        Ok(())
    }
}