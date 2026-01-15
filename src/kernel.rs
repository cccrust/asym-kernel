use crate::tensor::HyperTensor;
use crate::contract::Contract;
use crate::ops::Operator;
use anyhow::{Result, Context}; // 增加 Context 方便除錯
use std::collections::HashMap;
use std::sync::RwLock;
use std::path::PathBuf;
use std::fs;

pub struct NeuralKernel {
    pub device: candle_core::Device,
    pub memory: RwLock<HashMap<String, candle_core::Tensor>>,
    pub storage_dir: PathBuf, // 永續儲存目錄
}

impl NeuralKernel {
    pub fn new() -> Self {
        let storage_dir = PathBuf::from("asym_storage");
        if !storage_dir.exists() {
            fs::create_dir_all(&storage_dir).unwrap();
        }

        Self { 
            device: candle_core::Device::Cpu,
            memory: RwLock::new(HashMap::new()),
            storage_dir,
        }
    }

    fn run_step(&self, mut data: candle_core::Tensor, program: Vec<Operator>) -> Result<candle_core::Tensor> {
        for op in program {
            match op {
                Operator::Store(key) => {
                    // 1. 存入記憶體
                    {
                        let mut mem = self.memory.write().unwrap();
                        mem.insert(key.clone(), data.clone());
                    }

                    // 2. 存入磁碟 (使用 safetensors 格式)
                    let file_path = self.storage_dir.join(format!("{}.safetensors", key));
                    let mut map = HashMap::new();
                    map.insert(key.clone(), data.clone());
                    
                    // candle_core 提供直接存儲為 safetensors 的方法
                    candle_core::safetensors::save(&map, &file_path)
                        .context("Failed to persist tensor to disk")?;
                    
                    println!("    --> [Persistent Memory] Saved '{}' to {:?}", key, file_path);
                }

                Operator::Load(key) => {
                    // 1. 使用一個區塊(Scope)來確保讀取鎖會被自動釋放
                    let maybe_tensor = {
                        let mem_read = self.memory.read().unwrap();
                        mem_read.get(&key).cloned()
                    };

                    if let Some(saved_tensor) = maybe_tensor {
                        data = saved_tensor;
                        println!("    --> [Memory] Loaded '{}' from RAM", key);
                    } else {
                        // 2. 鎖已經釋放了，現在可以安全地去讀磁碟
                        let file_path = self.storage_dir.join(format!("{}.safetensors", key));
                        if file_path.exists() {
                            println!("    --> [Disk] Loading '{}' from {:?}", key, file_path);
                            
                            // 這裡會讀取整份 safetensors 檔案到 HashMap 中
                            let loaded_tensors = candle_core::safetensors::load(&file_path, &self.device)
                                .context("Failed to load safetensors from disk")?;
                            
                            // 修改點：使用 Some 而非 Ok，因為 HashMap.get 回傳的是 Option
                            if let Some(disk_tensor) = loaded_tensors.get(&key) {
                                data = disk_tensor.clone();
                                // 將磁碟讀出的張量寫回記憶體緩存
                                self.memory.write().unwrap().insert(key.clone(), data.clone());
                            } else {
                                anyhow::bail!("Memory Error: Key '{}' not found in file {:?}", key, file_path);
                            }
                        } else {
                            anyhow::bail!("Memory Error: Slot '{}' not found in RAM or Disk", key);
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
        // 初始驗證... (省略，同前)
        let final_data = self.run_step(input.data, program)?;
        let output = HyperTensor { data: final_data, label: "output".to_string() };
        contract.verify(&output)?;
        Ok(output)
    }
}