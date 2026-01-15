use crate::tensor::HyperTensor;
use anyhow::{Result, bail};
use std::sync::Arc; // 引入 Arc

pub struct Contract {
    pub expected_shape: Vec<usize>,
    // 將 Box 換成 Arc，這樣它就可以被輕鬆複製
    pub invariant: Arc<dyn Fn(&HyperTensor) -> bool + Send + Sync>,
}

impl Contract {
    pub fn verify(&self, tensor: &HyperTensor) -> Result<()> {
        if tensor.data.dims() != self.expected_shape {
            bail!("Dimension Mismatch: Expected {:?}, got {:?}", self.expected_shape, tensor.data.dims());
        }
        if !(self.invariant)(tensor) {
            bail!("Formal Verification Failed: Invariant violation.");
        }
        Ok(())
    }
}

// 手動實作 Clone
impl Clone for Contract {
    fn clone(&self) -> Self {
        Self {
            expected_shape: self.expected_shape.clone(),
            invariant: Arc::clone(&self.invariant),
        }
    }
}