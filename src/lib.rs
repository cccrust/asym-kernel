// 1. 聲明內部模組
// 這些名稱必須與你在 src/ 目錄下建立的檔名一致
pub mod tensor;
pub mod contract;
pub mod ops;
pub mod kernel;

// 2. 重新導出 (Public Re-exports)
// 這樣使用者只需要寫 `use asym_kernel::NeuralKernel` 
// 而不需要寫 `use asym_kernel::kernel::NeuralKernel`
pub use kernel::NeuralKernel;
pub use tensor::HyperTensor;
pub use contract::Contract;
pub use ops::Operator; // 假設你在 ops.rs 定義了 Operator 列舉

// 3. 定義全域錯誤處理 (使用 thiserror)
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ASymError {
    #[error("Dimension mismatch: expected {expected:?}, found {found:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("Formal verification failed: {0}")]
    VerificationFailed(String),

    #[error("Hardware execution error: {0}")]
    HardwareError(String),

    #[error("Contract violation: {0}")]
    ContractViolation(String),

    #[error("Unknown error occurred")]
    Unknown,
}

// 4. 定義一個「預導出」模組 (Prelude)
// 方便使用者一次性匯入所有常用組件
pub mod prelude {
    pub use crate::tensor::HyperTensor;
    pub use crate::contract::Contract;
    pub use crate::kernel::NeuralKernel;
    pub use crate::ASymError;
}