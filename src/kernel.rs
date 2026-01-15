use crate::tensor::HyperTensor;
use crate::contract::Contract;
use crate::ops::Operator;
use anyhow::Result;

pub struct NeuralKernel {
    pub device: candle_core::Device,
}

impl NeuralKernel {
    pub fn new() -> Self {
        Self { device: candle_core::Device::Cpu }
    }

    // 核心遞歸執行器
    fn run_step(&self, mut data: candle_core::Tensor, program: Vec<Operator>) -> Result<candle_core::Tensor> {
        for op in program {
            match op {
                Operator::Branch { threshold, true_path, false_path } => {
                    // 計算目前張量的平均值作為決策依據
                    let mean = data.mean_all()?.to_scalar::<f32>()?;
                    println!("--> [Decision] Current Mean: {:.4}, Threshold: {}", mean, threshold);

                    if mean > threshold {
                        println!("    Path: TRUE_PATH (Mean > Threshold)");
                        data = self.run_step(data, true_path)?;
                    } else {
                        println!("    Path: FALSE_PATH (Mean <= Threshold)");
                        data = self.run_step(data, false_path)?;
                    }
                }
                _ => {
                    println!("    Executing: {:?}", op);
                    data = op.apply(data)?;
                }
            }
        }
        Ok(data)
    }

    pub async fn execute(
        &self,
        input: HyperTensor,
        contract: Contract,
        program: Vec<Operator>,
    ) -> Result<HyperTensor> {
        // 1. 初始驗證 (只檢查形狀)
        if input.data.dims() != contract.expected_shape {
            anyhow::bail!("Input shape mismatch");
        }

        // 2. 遞歸執行指令流
        let final_data = self.run_step(input.data, program)?;

        // 3. 封裝輸出
        let output = HyperTensor {
            data: final_data,
            label: "output".to_string(),
        };

        // 4. 最終驗證 (包含數學不變量)
        contract.verify(&output)?;

        Ok(output)
    }
}