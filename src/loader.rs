use crate::ops::Operator;
use crate::contract::{Contract, Invariant};
use anyhow::{Result, Context, bail};

/// A-Sym 程式的完整封裝結構
pub struct ASymProgram {
    pub contract: Contract,
    pub logic: Vec<Operator>,
    /// 進化配置 (迭代次數 Epochs, 學習率 LR)
    pub evolve_config: Option<(i32, f64)>,
}

/// A-Sym 語言解析核心
pub fn parse(content: &str) -> Result<ASymProgram> {
    let mut expected_shape = vec![];
    let mut logic = vec![];
    let mut invariant = Invariant::AlwaysTrue;
    let mut evolve_config = None;
    
    // 狀態機變數，追蹤目前正在解析哪一個區塊
    let mut current_section = "";

    for (line_num, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        let idx = line_num + 1; // 用於錯誤訊息

        // 1. 跳過空行與註解
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // 2. 區塊切換檢查
        if line.starts_with("@C") {
            current_section = "contract";
            // 解析形狀，例如 @C [3] 或 @C [1, 512]
            let s = line.replace("@C", "")
                        .replace("[", "")
                        .replace("]", "");
            expected_shape = s.split(',')
                .map(|x| x.trim().parse::<usize>().with_context(|| format!("L{}: 無法解析形狀數值 '{}'", idx, x)))
                .collect::<Result<Vec<usize>>>()?;
            continue;
        } 
        
        if line.starts_with("@L") {
            current_section = "logic";
            continue;
        } 
        
        if line.starts_with("@V") {
            current_section = "verify";
            continue;
        } 
        
        if line.starts_with("@E") {
            current_section = "evolve";
            continue;
        }

        // 3. 根據目前所在區塊解析具體內容
        match current_section {
            "logic" => {
                if line == "MatrixSort" {
                    logic.push(Operator::MatrixSort);
                } else if line == "Normalize" {
                    logic.push(Operator::Normalize);
                } else if line == "Square" {
                    logic.push(Operator::Square);
                } else if line == "Softmax" {
                    logic.push(Operator::Softmax);
                } else if line.starts_with("AddScalar") {
                    let val_str = line.split(':').nth(1)
                        .with_context(|| format!("L{}: AddScalar 格式錯誤，應為 AddScalar: 10.0", idx))?;
                    let val = val_str.trim().parse::<f32>()
                        .with_context(|| format!("L{}: AddScalar 數值解析失敗", idx))?;
                    logic.push(Operator::AddScalar(val));
                } else if line.starts_with("AddParam") {
                    let key = line.split('"').nth(1)
                        .with_context(|| format!("L{}: AddParam 格式錯誤，應為 AddParam \"key\"", idx))?;
                    logic.push(Operator::AddParam(key.to_string()));
                } else if line.starts_with("Store") {
                    let key = line.split('"').nth(1)
                        .with_context(|| format!("L{}: Store 格式錯誤，應為 Store \"key\"", idx))?;
                    logic.push(Operator::Store(key.to_string()));
                } else if line.starts_with("Load") {
                    let key = line.split('"').nth(1)
                        .with_context(|| format!("L{}: Load 格式錯誤，應為 Load \"key\"", idx))?;
                    logic.push(Operator::Load(key.to_string()));
                }
            },

            "verify" => {
                if line == "IsMonotonic" {
                    invariant = Invariant::IsMonotonic;
                } else if line.starts_with("MeanEquals") {
                    let val_str = line.split(':').nth(1)
                        .with_context(|| format!("L{}: MeanEquals 格式錯誤，應為 MeanEquals: 0.0", idx))?;
                    let val = val_str.trim().parse::<f32>()
                        .with_context(|| format!("L{}: MeanEquals 數值解析失敗", idx))?;
                    invariant = Invariant::MeanEquals(val);
                }
            },

            "evolve" => {
                // 解析格式: Epochs: 10, LR: 0.5
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 2 {
                    let epochs_str = parts[0].split(':').nth(1)
                        .with_context(|| format!("L{}: Epochs 格式錯誤", idx))?;
                    let lr_str = parts[1].split(':').nth(1)
                        .with_context(|| format!("L{}: LR 格式錯誤", idx))?;
                    
                    let epochs = epochs_str.trim().parse::<i32>()?;
                    let lr = lr_str.trim().parse::<f64>()?;
                    evolve_config = Some((epochs, lr));
                }
            },

            _ => {} // 跳過未定義區塊的內容
        }
    }

    // 基本校驗：確保至少定義了形狀與邏輯
    if expected_shape.is_empty() {
        bail!("A-Sym 語法錯誤: 缺少必要的 @C (Contract) 區塊定義");
    }
    if logic.is_empty() {
        bail!("A-Sym 語法錯誤: 缺少必要的 @L (Logic) 算子定義");
    }

    Ok(ASymProgram {
        contract: Contract { expected_shape, invariant },
        logic,
        evolve_config,
    })
}