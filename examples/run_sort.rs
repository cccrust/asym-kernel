use asym_kernel::prelude::*;
use asym_kernel::ops::Operator;
use candle_core::{Device, Tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 初始化 Kernel (目前使用 CPU)
    let device = Device::Cpu;
    let kernel = NeuralKernel::new();

    // 2. 準備輸入數據：[1.0, 2.0, 3.0, 4.0, 5.0]
    let input_data = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &device)?;
    let input = HyperTensor {
        data: input_data,
        label: "test_input".to_string(),
    };

    // 3. 定義 AI 指令流 (Program)
    // 我們執行：平方 -> 加 10 -> 正規化 -> 最後用 Softmax 轉成機率分佈
    let program = vec![
        Operator::Square,
        Operator::AddScalar(10.0),
        Operator::Normalize,
        Operator::Softmax,
    ];

    // 4. 定義契約 (Contract)
    // 我們要求輸出必須跟輸入長度一樣 (5)，且總和必須約等於 1 (Softmax 的特性)
    // 修正後的契約：只在標籤為 "output" 時才檢查 Softmax 特性
    let my_contract = Contract {
        expected_shape: vec![5],
        invariant: Box::new(|t| {
            // 如果是輸入階段，我們只檢查形狀（直接回傳 true）
            if t.label == "test_input" {
                return true;
            }
            
            // 如果是輸出階段，檢查總和是否為 1
            if let Ok(sum) = t.data.sum_all() {
                if let Ok(sum_val) = sum.to_scalar::<f32>() {
                    return (sum_val - 1.0).abs() < 0.001;
                }
            }
            false
        }),
    };

    println!("--- A-Sym Runtime Start ---");
    println!("Input: [1.0, 2.0, 3.0, 4.0, 5.0]");

    match kernel.execute(input, my_contract, program).await {
        Ok(res) => {
            let out_vals = res.data.to_vec1::<f32>()?;
            println!("Final AI-Output (Probabilities): {:?}", out_vals);
            println!("Contract Verification: PASSED");
        },
        Err(e) => {
            println!("Kernel Panic: {}", e);
        }
    }

    Ok(())
}