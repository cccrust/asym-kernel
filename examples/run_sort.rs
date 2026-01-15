use asym_kernel::prelude::*;
use asym_kernel::ops::Operator;
use candle_core::{Device, Tensor};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let kernel = NeuralKernel::new();

    // 1. 準備數據，給予明確的標籤
    let input_data = Tensor::new(&[5.0f32, 1.0, 4.0, 2.0, 3.0], &device)?;
    let input = HyperTensor {
        data: input_data,
        label: "test_input".to_string(),
    };

    // 2. 指令流
    let program = vec![Operator::MatrixSort];

    // 3. 契約：使用更寬鬆的標籤檢查，但嚴格的輸出檢查
    let sort_contract = Contract {
        expected_shape: vec![5],
        invariant: Arc::new(|t: &HyperTensor| {
            // 除錯：印出目前的標籤，看看 Kernel 到底傳了什麼進來
            // println!("DEBUG: Inspecting tensor with label '{}'", t.label);

            // 如果標籤包含 "input"，則這是 Pre-check，我們放行
            if t.label.contains("input") {
                return true;
            }
            
            // 如果標籤是 "output"，這是 Post-check，必須排序
            if t.label == "output" {
                if let Ok(vals) = t.data.to_vec1::<f32>() {
                    let is_sorted = vals.windows(2).all(|w| w[0] <= w[1]);
                    if !is_sorted {
                        println!("  [Contract Violation] Output not sorted: {:?}", vals);
                    }
                    return is_sorted;
                }
            }
            
            // 其他情況預設放行
            true
        }),
    };

    println!("--- A-Sym Matrix Sort Test ---");
    println!("Input (Unsorted): [5.0, 1.0, 4.0, 2.0, 3.0]");

    // 4. 執行
    match kernel.execute(input, sort_contract, program).await {
        Ok(res) => {
            let out_vals = res.data.to_vec1::<f32>()?;
            println!("Output (Sorted):   {:?}", out_vals);
            println!("Verification: PASSED");
        },
        Err(e) => {
            println!("Kernel Blocked: {}", e);
        }
    }

    Ok(())
}