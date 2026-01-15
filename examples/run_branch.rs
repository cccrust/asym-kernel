use asym_kernel::prelude::*;
use asym_kernel::ops::Operator;
use candle_core::{Device, Tensor};
use std::sync::Arc; // 引入 Arc

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let kernel = NeuralKernel::new();

    let ai_program = vec![
        Operator::Branch {
            threshold: 50.0,
            true_path: vec![Operator::Normalize],
            false_path: vec![Operator::Square, Operator::MatrixSort],
        }
    ];

    let contract = Contract {
        expected_shape: vec![3],
        // 改用 Arc::new
        invariant: Arc::new(|_t| true),
    };

    println!("--- Scenario 1: High Mean Data (100, 200, 300) ---");
    let input_high = HyperTensor {
        data: Tensor::new(&[100.0f32, 200.0, 300.0], &device)?,
        label: "input".to_string(),
    };
    // 現在可以 .clone() 了！
    let out_high = kernel.execute(input_high, contract.clone(), ai_program.clone()).await?;
    println!("Final High Result: {:?}", out_high.data.to_vec1::<f32>()?);

    println!("\n--- Scenario 2: Low Mean Data (1, 3, 2) ---");
    let input_low = HyperTensor {
        data: Tensor::new(&[1.0f32, 3.0, 2.0], &device)?,
        label: "input".to_string(),
    };
    let out_low = kernel.execute(input_low, contract, ai_program).await?;
    println!("Final Low Result: {:?}", out_low.data.to_vec1::<f32>()?);

    Ok(())
}