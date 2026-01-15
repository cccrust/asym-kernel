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

    pub async fn execute(
        &self,
        input: HyperTensor,
        contract: Contract,
        program: Vec<Operator>, // 這裡就是 AI 生成的指令序列
    ) -> Result<HyperTensor> {
        
        // 1. Pre-check
        contract.verify(&input)?;

        // 2. 算子編織與執行 (Operator Weaving)
        let mut current_data = input.data;
        for op in program {
            println!("Executing Op: {:?}", op);
            current_data = op.apply(current_data)?;
        }

        let output = HyperTensor {
            data: current_data,
            label: "output".to_string(),
        };

        // 3. Post-check (驗證輸出是否符合契約)
        contract.verify(&output)?;

        Ok(output)
    }
}