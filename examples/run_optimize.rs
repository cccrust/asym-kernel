use asym_kernel::prelude::*;
use asym_kernel::ops::Operator;
use candle_core::{Device, Tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let kernel = NeuralKernel::new();

    // 1. 註冊一個參數 "bias"，初始值為 10.0
    kernel.register_param("bias", 10.0)?;
    
    let input_data = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
    let program = vec![Operator::AddParam("bias".to_string())];

    println!("--- Initial State ---");
    println!("Target Mean: 0.0");
    
    // 2. 進行多次「自我修正」迭代
    for i in 1..=10 {
        let loss = kernel.train_step(input_data.clone(), program.clone(), 0.0, 0.5).await?;
        
        let current_bias = kernel.params.read().unwrap()
            .get("bias").unwrap().as_tensor().to_scalar::<f32>()?;
            
        println!("Iteration {}: Loss = {:.4}, Bias = {:.4}", i, loss, current_bias);
        
        if loss < 0.0001 {
            println!("--- Goal Reached! Code has evolved to meet the Contract. ---");
            break;
        }
    }

    Ok(())
}