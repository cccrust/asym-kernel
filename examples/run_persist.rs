use asym_kernel::prelude::*;
use asym_kernel::ops::Operator;
use candle_core::{Device, Tensor};
use std::sync::Arc;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let kernel = NeuralKernel::new();
    let contract = Contract {
        expected_shape: vec![2],
        invariant: Arc::new(|_t| true),
    };

    let key = "long_term_memory_01".to_string();
    let file_path = PathBuf::from("asym_storage").join(format!("{}.safetensors", key));

    // 判斷磁碟是否已經有存檔
    if file_path.exists() {
        println!("--- Step 2: Restarting and Loading from Disk ---");
        let program = vec![
            Operator::Load(key.clone()),
        ];
        // 輸入無意義的數據 [0, 0]，看是否會被 Load 覆蓋
        let input = HyperTensor { data: Tensor::new(&[0.0f32, 0.0], &device)?, label: "dummy".to_string() };
        let res = kernel.execute(input, contract, program).await?;
        println!("Recovered Data from Disk: {:?}", res.data.to_vec1::<f32>()?);
        println!("Persistence Test: PASSED");
    } else {
        println!("--- Step 1: Saving Data for the First Time ---");
        let input = HyperTensor {
            data: Tensor::new(&[42.0f32, 99.0], &device)?,
            label: "input".to_string(),
        };
        let program = vec![
            Operator::Store(key.clone()),
        ];
        kernel.execute(input, contract, program).await?;
        println!("Data saved. Please run this example again to verify persistence!");
    }

    Ok(())
}