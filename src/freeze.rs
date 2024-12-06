//! A helper to freeze tensors/layers.

/// A helper trait to freeze/unfreeze the weights.
pub trait Freeze {
    /// Freezes or unfreezes all the tensors inside the module.
    fn set_freeze(&self, freeze: bool);

    /// Freezes all the tensors inside the module.
    fn freeze(&self) {
        self.set_freeze(true)
    }

    /// Unfreezes all the tensors inside the module.
    fn unfreeze(&self) {
        self.set_freeze(false)
    }
}

impl<T: Freeze> Freeze for Option<T> {
    fn set_freeze(&self, freeze: bool) {
        if let Some(this) = self {
            this.set_freeze(freeze)
        }
    }
}

impl Freeze for tch::Tensor {
    fn set_freeze(&self, freeze: bool) {
        let _ = tch::Tensor::set_requires_grad(self, !freeze);
    }
}

impl Freeze for tch::nn::Linear {
    fn set_freeze(&self, freeze: bool) {
        self.ws.set_freeze(freeze);
        self.bs.set_freeze(freeze);
    }
}

impl Freeze for tch::nn::LayerNorm {
    fn set_freeze(&self, freeze: bool) {
        self.bs.set_freeze(freeze);
        self.ws.set_freeze(freeze);
    }
}

impl Freeze for tch::nn::Embedding {
    fn set_freeze(&self, freeze: bool) {
        self.ws.set_freeze(freeze)
    }
}
