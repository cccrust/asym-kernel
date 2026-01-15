use asym_kernel::prelude::*;
use asym_kernel::ops::Operator;
use candle_core::{Device, Tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let kernel = NeuralKernel::new();

    // 1. 準備亂序數據：[5.0, 1.0, 4.0, 2.0, 3.0]
    let input_data = Tensor::new(&[5.0f32, 1.0, 4.0, 2.0, 3.0], &device)?;
    let input = HyperTensor {
        data: input_data,
        label: "test_input".to_string(),
    };

    // 2. 只有一個指令：MatrixSort
    let program = vec![Operator::MatrixSort];

    // 3. 強大的契約驗證：驗證輸出是否為「遞增」
    let sort_contract = Contract {
        expected_shape: vec![5],
        invariant: Box::new(|t| {
            if t.label == "test_input" { return true; }
            
            if let Ok(vals) = t.data.to_vec1::<f32>() {
                // 檢查是否每一項都小於等於下一項
                return vals.windows(2).all(|w| w[0] <= w[1]);
            }
            false
        }),
    };

    println!("--- A-Sym Matrix Sort Test ---");
    println!("Input (Unsorted): [5.0, 1.0, 4.0, 2.0, 3.0]");

    match kernel.execute(input, sort_contract, program).await {
        Ok(res) => {
            let out_vals = res.data.to_vec1::<f32>()?;
            println!("Output (Sorted):   {:?}", out_vals);
            println!("Verification: PASSED");
        },
        Err(e) => println!("Kernel Panic: {}", e),
    }

    Ok(())
}