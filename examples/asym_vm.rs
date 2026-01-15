// examples/asym_vm.rs
use asym_kernel::loader;
use asym_kernel::kernel::NeuralKernel;
use asym_kernel::tensor::HyperTensor;
use candle_core::{Device, Tensor};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 從命令行獲取文件名
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --example asym_vm -- <file.asym>");
        return Ok(());
    }
    let filename = &args[1];

    // 2. 初始化環境
    let kernel = NeuralKernel::new();
    let device = Device::Cpu;

    // 3. 讀取並解析 .asym 語言
    println!("--- Loading A-Sym Program: {} ---", filename);
    let code = std::fs::read_to_string(filename)?;
    let program = loader::parse(&code)?;

    // 4. 準備測試數據 (根據 C 區塊動態調整)
    let shape = &program.contract.expected_shape;
    let raw_data = if shape[0] == 5 {
        vec![5.0f32, 1.0, 4.0, 2.0, 3.0]
    } else {
        vec![1.0f32, 2.0, 3.0]
    };
    
    let input = HyperTensor {
        data: Tensor::new(raw_data.as_slice(), &device)?,
        label: "test_input".to_string(),
    };

    // 5. 虛擬機執行
    match kernel.execute(input, program.contract, program.logic).await {
        Ok(res) => {
            println!("VM Execution Success!");
            println!("Final Tensor: {:?}", res.data.to_vec1::<f32>()?);
        }
        Err(e) => println!("VM Panic: {}", e),
    }

    Ok(())
}