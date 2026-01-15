use asym_kernel::loader;
use asym_kernel::kernel::NeuralKernel;
use asym_kernel::tensor::HyperTensor;
use asym_kernel::contract::Invariant;
use candle_core::{Device, Tensor};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 取得指令列參數 (例如: cargo run --example asym_vm -- run_evolve.asym)
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("用法: cargo run --example asym_vm -- <檔案名稱.asym>");
        return Ok(());
    }
    let filename = &args[1];

    // 2. 初始化 VM 環境 (Kernel)
    let device = Device::Cpu;
    let kernel = NeuralKernel::new();

    // 3. 讀取並解析 .asym 原始碼
    println!("--- [A-Sym VM] 正在加載程式: {} ---", filename);
    let code = std::fs::read_to_string(filename)?;
    let program = loader::parse(&code)?;

    // 4. 準備測試數據
    // 我們根據 @C 定義的形狀自動生成數據 [1.0, 2.0, 3.0, ...]
    let shape = &program.contract.expected_shape;
    let size: usize = shape.iter().product();
    let raw_data: Vec<f32> = (1..=size).map(|v| v as f32).collect();
    
    let input_tensor = Tensor::new(raw_data.as_slice(), &device)?;
    
    // 如果程式中有用到 "bias" 參數，我們預設給它一個錯誤的初始值 10.0，引發進化
    // 在更進階的系統中，這可以透過 @P 區塊來定義初始值
    let _ = kernel.register_param("bias", 10.0);

    // 5. 嘗試標準執行模式
    println!("--- [A-Sym VM] 進入標準執行模式 ---");
    let input = HyperTensor {
        data: input_tensor.clone(),
        label: "test_input".to_string(),
    };

    match kernel.execute(input.clone(), program.contract.clone(), program.logic.clone()).await {
        Ok(res) => {
            println!("--- [A-Sym VM] 標準執行成功! ---");
            println!("最終結果: {:?}", res.data.to_vec1::<f32>()?);
        }
        Err(e) => {
            println!("--- [A-Sym VM] 標準執行失敗: {} ---", e);

            // 6. 檢查是否具備「自我進化」條件
            if let Some((epochs, lr)) = program.evolve_config {
                if let Invariant::MeanEquals(target) = program.contract.invariant {
                    println!("--- [A-Sym VM] 檢測到進化指令 (@E)，啟動自我修正機制 ---");
                    println!("目標平均值: {}, 預期迭代: {} 次, 學習率: {}", target, epochs, lr);

                    for i in 1..=epochs {
                        let loss = kernel.train_step(
                            input_tensor.clone(), 
                            program.logic.clone(), 
                            target, 
                            lr
                        ).await?;
                        
                        // 取得目前的參數狀態
                        let current_bias = kernel.params.read().unwrap()
                            .get("bias")
                            .map(|v| v.as_tensor().to_scalar::<f32>().unwrap_or(0.0))
                            .unwrap_or(0.0);

                        println!("  迭代 {:02}/{}: Loss = {:.6}, Bias = {:.4}", i, epochs, loss, current_bias);
                        
                        if loss < 0.00001 {
                            println!("--- [A-Sym VM] 邏輯已收斂，達成契約要求 ---");
                            break; 
                        }
                    }

                    // 7. 進化結束後再次執行，驗證重生後的程式碼
                    println!("--- [A-Sym VM] 重新執行驗證 ---");
                    match kernel.execute(input, program.contract, program.logic).await {
                        Ok(res) => {
                            println!("--- [A-Sym VM] 重生成功! ---");
                            println!("最終結果: {:?}", res.data.to_vec1::<f32>()?);
                            println!("狀態: 程式碼已完成自我進化 (EVOLVED)");
                        },
                        Err(final_err) => println!("進化失敗: {}", final_err),
                    }
                } else {
                    println!("錯誤: 目前進化模式僅支援 MeanEquals 類型的不變量。");
                }
            } else {
                println!("提示: 程式碼中未定義 @E 區塊，無法進行自動修正。");
            }
        }
    }

    Ok(())
}