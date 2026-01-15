use asym_kernel::prelude::*;
use asym_kernel::ops::Operator;
use candle_core::{Device, Tensor};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let kernel = NeuralKernel::new(); // 同一個 Kernel 實例持有同一個記憶體池
    let contract = Contract {
        expected_shape: vec![3],
        invariant: Arc::new(|_t| true),
    };

    // 任務 A: 儲存邏輯
    println!("--- Task A: Process and Store ---");
    let input_a = HyperTensor {
        data: Tensor::new(&[1.0f32, 2.0, 3.0], &device)?,
        label: "input".to_string(),
    };
    let program_a = vec![
        Operator::Square,
        Operator::Store("feature_01".to_string()),
    ];
    kernel.execute(input_a, contract.clone(), program_a).await?;

    // 任務 B: 讀取邏輯
    println!("\n--- Task B: Load and Finalize ---");
    // 即使輸入是 0，我們也會因為 Load 而覆蓋它
    let input_b = HyperTensor {
        data: Tensor::new(&[0.0f32, 0.0, 0.0], &device)?,
        label: "dummy_input".to_string(),
    };
    let program_b = vec![
        Operator::Load("feature_01".to_string()),
        Operator::Normalize,
    ];
    
    let result = kernel.execute(input_b, contract, program_b).await?;
    println!("Final Result from Memory: {:?}", result.data.to_vec1::<f32>()?);

    Ok(())
}