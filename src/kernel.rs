use crate::tensor::HyperTensor;
use crate::contract::Contract;
use crate::ops::Operator;
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::RwLock;
use std::path::PathBuf;
use candle_core::{Tensor, Var, Device};

pub struct NeuralKernel {
    pub device: Device,
    pub memory: RwLock<HashMap<String, Tensor>>,
    pub params: RwLock<HashMap<String, Var>>, 
    pub storage_dir: PathBuf,
}

impl NeuralKernel {
    pub fn new() -> Self {
        let storage_dir = PathBuf::from("asym_storage");
        if !storage_dir.exists() {
            std::fs::create_dir_all(&storage_dir).unwrap();
        }
        Self { 
            device: Device::Cpu,
            memory: RwLock::new(HashMap::new()),
            params: RwLock::new(HashMap::new()),
            storage_dir,
        }
    }

    pub fn register_param(&self, name: &str, initial_value: f32) -> Result<()> {
        let tensor = Tensor::new(initial_value, &self.device)?;
        let var = Var::from_tensor(&tensor)?;
        self.params.write().unwrap().insert(name.to_string(), var);
        Ok(())
    }

    fn run_step(&self, mut data: Tensor, program: Vec<Operator>) -> Result<Tensor> {
        for op in program {
            match op {
                Operator::AddParam(name) => {
                    let params = self.params.read().unwrap();
                    let p = params.get(&name).context("Param not found")?;
                    data = data.broadcast_add(p.as_tensor())?;
                }
                Operator::Store(key) => {
                    // 1. 存入記憶體
                    self.memory.write().unwrap().insert(key.clone(), data.clone());
                    
                    // 2. 存入磁碟 (修正點：使用 Owned Tensor)
                    let file_path = self.storage_dir.join(format!("{}.safetensors", key));
                    let mut map = HashMap::new();
                    // 這裡 key 和 data 都用 clone()，把所有權交給 map
                    map.insert(key.clone(), data.clone());
                    
                    candle_core::safetensors::save(&map, &file_path)
                        .context("Failed to save safetensors")?;
                    println!("    --> [Persistent Memory] Saved '{}' to disk", key);
                }
                Operator::Load(key) => {
                    let maybe_tensor = {
                        let mem_read = self.memory.read().unwrap();
                        mem_read.get(&key).cloned()
                    };

                    if let Some(t) = maybe_tensor {
                        data = t;
                        println!("    --> [Memory] Loaded '{}' from RAM", key);
                    } else {
                        let file_path = self.storage_dir.join(format!("{}.safetensors", key));
                        if file_path.exists() {
                            let loaded = candle_core::safetensors::load(&file_path, &self.device)
                                .context("Failed to load safetensors")?;
                            if let Some(t) = loaded.get(&key) {
                                data = t.clone();
                                self.memory.write().unwrap().insert(key.clone(), data.clone());
                                println!("    --> [Disk] Loaded '{}' from disk", key);
                            }
                        } else {
                            anyhow::bail!("Memory Error: Slot '{}' not found", key);
                        }
                    }
                }
                Operator::Branch { threshold, true_path, false_path } => {
                    let mean = data.mean_all()?.to_scalar::<f32>()?;
                    if mean > threshold {
                        data = self.run_step(data, true_path)?;
                    } else {
                        data = self.run_step(data, false_path)?;
                    }
                }
                _ => {
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
        contract.verify(&input)?; 
        let final_data = self.run_step(input.data, program)?;
        let output = HyperTensor { data: final_data, label: "output".to_string() };
        contract.verify(&output)?;
        Ok(output)
    }

    pub async fn train_step(
        &self,
        input: Tensor,
        program: Vec<Operator>,
        target_mean: f32,
        learning_rate: f64,
    ) -> Result<f32> {
        let output = self.run_step(input, program)?;
        let mean = output.mean_all()?;
        let target = Tensor::new(target_mean, &self.device)?;
        let loss = mean.sub(&target)?.sqr()?;
        let grads = loss.backward()?;
        
        let mut params = self.params.write().unwrap();
        for var in params.values_mut() {
            if let Some(grad) = grads.get(var) {
                let lr_tensor = Tensor::new(learning_rate as f32, &self.device)?;
                let updated = var.as_tensor().sub(&(grad * lr_tensor)?)?;
                var.set(&updated)?;
            }
        }
        Ok(loss.to_scalar::<f32>()?)
    }
}