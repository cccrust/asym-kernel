use crate::tensor::HyperTensor;
use anyhow::{Result, bail};

pub struct Contract {
    pub expected_shape: Vec<usize>,
    pub invariant: Box<dyn Fn(&HyperTensor) -> bool + Send + Sync>,
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