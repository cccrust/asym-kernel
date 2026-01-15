use crate::tensor::HyperTensor;
use crate::contract::Contract;
use crate::ops::Operator;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::RwLock;

pub struct NeuralKernel {
    pub device: candle_core::Device,
    // 持久化記憶體池：Key 為槽位名稱，Value 為張量
    pub memory: RwLock<HashMap<String, candle_core::Tensor>>,
}

impl NeuralKernel {
    pub fn new() -> Self {
        Self { 
            device: candle_core::Device::Cpu,
            memory: RwLock::new(HashMap::new()),
        }
    }

    fn run_step(&self, mut data: candle_core::Tensor, program: Vec<Operator>) -> Result<candle_core::Tensor> {
        for op in program {
            match op {
                Operator::Store(key) => {
                    let mut mem = self.memory.write().unwrap();
                    mem.insert(key.clone(), data.clone());
                    println!("    --> [Memory] Stored tensor to slot: '{}'", key);
                }
                Operator::Load(key) => {
                    let mem = self.memory.read().unwrap();
                    if let Some(saved_tensor) = mem.get(&key) {
                        data = saved_tensor.clone();
                        println!("    --> [Memory] Loaded tensor from slot: '{}'", key);
                    } else {
                        anyhow::bail!("Memory Error: Slot '{}' is empty", key);
                    }
                }
                Operator::Branch { threshold, true_path, false_path } => {
                    let mean = data.mean_all()?.to_scalar::<f32>()?;
                    println!("--> [Decision] Mean: {:.4}, Threshold: {}", mean, threshold);
                    if mean > threshold {
                        data = self.run_step(data, true_path)?;
                    } else {
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
        if input.data.dims() != contract.expected_shape {
            anyhow::bail!("Input shape mismatch");
        }

        let final_data = self.run_step(input.data, program)?;

        let output = HyperTensor {
            data: final_data,
            label: "output".to_string(),
        };

        contract.verify(&output)?;
        Ok(output)
    }
}