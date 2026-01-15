use crate::ops::Operator;
use crate::contract::{Contract, Invariant};
use anyhow::{Result, Context};

pub struct ASymProgram {
    pub contract: Contract,
    pub logic: Vec<Operator>,
}

pub fn parse(content: &str) -> Result<ASymProgram> {
    let mut expected_shape = vec![];
    let mut logic = vec![];
    let mut invariant = Invariant::AlwaysTrue;
    let mut current_section = "";

    for line in content.lines() {
        let line = line.trim();
        // 跳過空行和註釋
        if line.is_empty() || line.starts_with('#') { continue; }

        if line.starts_with("@C") {
            // 解析形狀: @C [5]
            let s = line.replace("@C", "")
                        .replace("[", "")
                        .replace("]", "");
            expected_shape = s.split(',')
                .map(|x| x.trim().parse::<usize>().context("Invalid shape number"))
                .collect::<Result<Vec<usize>>>()?;
        } else if line.starts_with("@L") {
            current_section = "logic";
        } else if line.starts_with("@V") {
            current_section = "verify";
        } else if current_section == "logic" {
            // 算子解析
            if line == "MatrixSort" { 
                logic.push(Operator::MatrixSort); 
            } else if line == "Normalize" { 
                logic.push(Operator::Normalize); 
            } else if line == "Square" { 
                logic.push(Operator::Square); 
            } else if line.starts_with("AddScalar") {
                // 修正點：先用 ? 處理 Option，再 trim()
                let val_str = line.split(':').nth(1)
                    .context("AddScalar requires a value, e.g., 'AddScalar: 10.0'")?;
                let val = val_str.trim().parse::<f32>()?;
                logic.push(Operator::AddScalar(val));
            } else if line.starts_with("Store") {
                let key = line.split('"').nth(1)
                    .context("Store requires a key in quotes, e.g., Store \"my_key\"")?;
                logic.push(Operator::Store(key.to_string()));
            } else if line.starts_with("Load") {
                let key = line.split('"').nth(1)
                    .context("Load requires a key in quotes, e.g., Load \"my_key\"")?;
                logic.push(Operator::Load(key.to_string()));
            }
        } else if current_section == "verify" {
            // 驗證器解析
            if line == "IsMonotonic" { 
                invariant = Invariant::IsMonotonic; 
            } else if line.starts_with("MeanEquals") {
                // 同樣的修正點
                let val_str = line.split(':').nth(1)
                    .context("MeanEquals requires a value, e.g., 'MeanEquals: 0.0'")?;
                let val = val_str.trim().parse::<f32>()?;
                invariant = Invariant::MeanEquals(val);
            }
        }
    }

    // 檢查是否有解析到形狀
    if expected_shape.is_empty() {
        anyhow::bail!("A-Sym Program Error: Missing @C section");
    }

    Ok(ASymProgram {
        contract: Contract { expected_shape, invariant },
        logic,
    })
}