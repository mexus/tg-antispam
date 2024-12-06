//! Optimizer wrapper.

use std::{num::NonZeroUsize, sync::Arc};

use tch::Tensor;

/// A wrapper around the torch's optimizer with a certain learning rate
/// strategy.
pub struct Optimizer {
    /// Trainable variables.
    variables: std::sync::Arc<std::sync::Mutex<tch::nn::Variables>>,

    /// Optimizer configuration.
    cfg: OptimizerConfiguration,

    /// Current epoch.
    current_epoch: usize,

    /// Number of group of tensors of the underlying BERT model.
    bert_tensors_group: usize,

    /// The amount of steps in the warmup phase.
    warmup_steps: usize,

    /// The amount of steps after the warmup phase.
    normal_steps: usize,

    /// Size of the macro-batch. Defines for how long gradients are accumulated.
    macro_batch: usize,
}

/// Optimizer configuration.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct OptimizerConfiguration {
    /// How learning rate changes within the epoch.
    pub within_epoch: LearningRateScheduler,

    /// How learning rate changes between the epochs.
    pub inter_epoch: EpochLearningRate,

    /// The minimum learning rate in the warmup phase.
    pub warmup: f64,

    /// Applies warmup for this proportion of each training data (0.0 to 1.0).
    #[serde(rename = "warmup_period")]
    pub warmup_steps: f64,

    /// Pretrained (BERT) layers will have its learning rate multiplied by this
    /// value.
    ///
    /// This value should normally be lesser than 1 but remain positive.
    pub bert_multiplier: f64,
}

/// How learning rate changes from epoch to epoch.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpochLearningRate {
    /// Do not modify base learning rate from epoch to epoch.
    Constant {
        /// Base learning rate. Can be modified by the learning rate scheduler
        /// within the epoch.
        base: f64,
    },
    /// The learning rate decreases linearly
    LinearDecrease {
        /// Base learning rate. Can be modified by the learning rate scheduler
        /// within the epoch.
        base: f64,
        /// The learning rate is decreased by this amount each epoch.
        delta: f64,
        /// The learning rate won't go below this value.
        min: f64,
    },
    /// The learning rate decreases exponentially.
    ExponentDecrease {
        /// Base learning rate. Can be modified by the learning rate scheduler
        /// within the epoch.
        base: f64,
        /// Every next epoch will have its base learning rate divided by this
        /// factor.
        factor: f64,
        /// The learning rate won't go below this value.
        min: f64,
    },
}

/// Learning rate scheduler variants.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningRateScheduler {
    /// Cosine annealing with warm restarts.
    CosineAnnealing {
        /// The amount of restarts within the epoch. Set to zero for no restarts
        /// at all.
        restarts: usize,
        /// The factor by which the learning rate is decreased during the
        /// training.
        factor: f64,
    },
    /// The learning rate doesn't change within the epoch.
    Constant,
}

/// Optimizer parameters for a single epoch.
pub struct EpochOptimizer<'a> {
    inner: tch::COptimizer,

    variables: &'a std::sync::Mutex<tch::nn::Variables>,

    /// Base learning rate for the current epoch.
    base_learning_rate: f64,

    /// Current step within the epoch.
    current_step: usize,

    /// Number of group of tensors of the underlying BERT model.
    bert_tensors_group: usize,

    /// Last set learning rate (for the "upper" layer).
    last_lr: f64,

    /// Warm up minimum learning rate.
    warmup_lr: f64,

    /// The amount of steps in the warmup phase.
    warmup_steps: usize,
    /// The amount of steps after the warmup phase.
    normal_steps: usize,

    /// Size of the macro-batch. Defines for how long gradients are accumulated.
    macro_batch: usize,

    /// The amount of processed mini batches.
    mini_batches_processed: usize,

    /// Learning rate scheduler.
    scheduler: &'a LearningRateScheduler,

    /// A learning rate multiplier for the BERT layers.
    bert_multiplier: f64,
}

impl<'a> EpochOptimizer<'a> {
    /// Returns the base learning rate for the current epoch.
    pub fn base_learning_rate(&self) -> f64 {
        self.base_learning_rate
    }

    /// Finishes the epoch explicitly.
    pub fn finish(mut self) {
        self.finish_impl();
    }

    fn finish_impl(&mut self) {
        if self.mini_batches_processed != 0 {
            // Perform the gradient descend.
            self.inner.step().expect("Gradient descend failed");
        }
        self.mini_batches_processed = 0;
    }

    /// Computes the learning rate for the current step.
    fn compute_lr(&self) -> f64 {
        let epoch_lr = self.base_learning_rate;

        if self.current_step < self.warmup_steps {
            // Warmup phase.
            let factor =
                self.current_step as f64 / self.warmup_steps.saturating_sub(1).max(1) as f64;
            (epoch_lr - self.warmup_lr) * factor + self.warmup_lr
        } else {
            // Normal phase.

            // Adjust the step number.
            let step = self.current_step - self.warmup_steps;
            match self.scheduler {
                LearningRateScheduler::CosineAnnealing { restarts, factor } => {
                    let restart_every = self.normal_steps / (restarts + 1);
                    let progress = step as f64 / restart_every as f64 % 1.;

                    let min_lr = epoch_lr / factor;
                    let max_lr = epoch_lr;

                    let annealing = 0.5 + 0.5 * (progress * std::f64::consts::PI).cos();
                    min_lr + annealing * (max_lr - min_lr)
                }
                LearningRateScheduler::Constant => epoch_lr,
            }
        }
    }

    fn reset_when_required(&mut self) {
        if self.current_step < self.warmup_steps {
            return;
        }
        let step = self.current_step - self.warmup_steps;
        match self.scheduler {
            LearningRateScheduler::CosineAnnealing { restarts, .. } => {
                let restart_every = self.normal_steps / (restarts + 1);
                if step % restart_every == 0 {
                    // Reset momentum.
                    self.reset_inner();
                }
            }
            _ => { /* No op */ }
        }
    }

    fn set_lr(&mut self, learning_rate: f64) {
        self.last_lr = learning_rate;
        self.inner
            .set_learning_rate(learning_rate)
            .expect("We expect this to always work");

        if self.bert_tensors_group != 0 {
            self.inner
                .set_learning_rate_group(
                    self.bert_tensors_group,
                    learning_rate * self.bert_multiplier,
                )
                .expect("We expect this to always work");
        }
    }

    /// Returns the current learning rate for the upper layer.
    pub fn learning_rate(&self) -> f64 {
        self.last_lr
    }

    /// Processes the mini batch.
    pub fn next_mini_batch(&mut self, loss: &Tensor) {
        let lr = self.compute_lr();

        self.current_step += 1;
        self.set_lr(lr);

        // Compute (and accumulate) gradients.
        loss.backward();

        self.mini_batches_processed += 1;
        if self.mini_batches_processed == self.macro_batch {
            self.mini_batches_processed = 0;

            self.reset_when_required();

            // Perform the gradient descend for the accumulated gradients.
            self.inner.step().expect("Gradient descend failed");
            self.inner.zero_grad().expect("Can't reset gradients");
        }
    }

    /// Resets internal optimizer.
    pub fn reset_inner(&mut self) {
        self.inner = prepare_c_optimizer(self.variables);
    }
}

impl Drop for EpochOptimizer<'_> {
    fn drop(&mut self) {
        self.finish_impl();
    }
}

impl Optimizer {
    /// Initializes a new optimizer with the learning rate strategy.
    ///
    /// `total_steps` represents the amount of mini batches.
    pub fn new(
        var_store: &tch::nn::VarStore,
        cfg: OptimizerConfiguration,
        total_steps: usize,
        bert_tensors_group: Option<NonZeroUsize>,
        macro_batch: usize,
    ) -> Self {
        let variables = Arc::clone(&var_store.variables_);

        let warmup_steps = cfg.warmup_steps.clamp(0., 1.) * total_steps as f64;
        let warmup_steps = (warmup_steps.round() as usize).min(total_steps);
        let normal_steps = total_steps - warmup_steps;

        let macro_batch = macro_batch.max(1);

        match cfg.within_epoch {
            LearningRateScheduler::CosineAnnealing { factor, .. } => {
                assert!(
                    factor >= 1.,
                    "Cosine annealing factor must be greater than or equal to 1"
                );
            }
            LearningRateScheduler::Constant => { /* No op */ }
        }

        match cfg.inter_epoch {
            EpochLearningRate::Constant { .. } => { /* No op */ }
            EpochLearningRate::LinearDecrease { base, delta, min } => {
                assert!(
                    base > min,
                    "Base learning rate must be greater than the minimal"
                );
                assert!(delta > 0., "Learning rate delta must be greater than zero");
                let epochs = (base - min) / delta;
                let epochs = epochs.ceil();
                eprintln!(
                    "The minimum base learning rate will be reached in \
                     {epochs} epochs"
                );
            }
            EpochLearningRate::ExponentDecrease { base, factor, min } => {
                assert!(
                    base > min,
                    "Base learning rate must be greater than the minimal"
                );
                assert!(
                    factor >= 1.0,
                    "Epoch learning rate exponential reduce factor must \
                     greater than or equal to 1"
                );
                let epochs = (base / min).log(factor);
                let epochs = epochs.ceil();
                eprintln!(
                    "The minimum base learning rate will be reached in \
                     {epochs} epochs"
                );
            }
        }

        Self {
            variables,
            cfg,
            current_epoch: 0,
            bert_tensors_group: bert_tensors_group.map(NonZeroUsize::get).unwrap_or(0),
            warmup_steps,
            normal_steps,
            macro_batch,
        }
    }

    /// Starts the epoch.
    pub fn start_epoch(&mut self) -> EpochOptimizer<'_> {
        let base_learning_rate = match self.cfg.inter_epoch {
            EpochLearningRate::Constant { base } => base,
            EpochLearningRate::LinearDecrease { base, delta, min } => {
                (base - delta * (self.current_epoch as f64)).max(min)
            }
            EpochLearningRate::ExponentDecrease { base, factor, min } => {
                (base / factor.powi(self.current_epoch as i32)).max(min)
            }
        };
        self.current_epoch += 1;

        EpochOptimizer {
            inner: prepare_c_optimizer(&self.variables),
            variables: &self.variables,
            base_learning_rate,
            current_step: 0,
            bert_tensors_group: self.bert_tensors_group,
            last_lr: self.cfg.warmup,
            warmup_lr: self.cfg.warmup,
            warmup_steps: self.warmup_steps,
            normal_steps: self.normal_steps,
            macro_batch: self.macro_batch,
            mini_batches_processed: 0,
            scheduler: &self.cfg.within_epoch,
            bert_multiplier: self.cfg.bert_multiplier,
        }
    }
}

fn prepare_c_optimizer(variables: &std::sync::Mutex<tch::nn::Variables>) -> tch::COptimizer {
    // Default AdamW parameters.
    let beta1 = 0.9;
    let beta2 = 0.999;
    let wd = 0.01;
    let eps = 1e-8;
    let amsgrad = false;

    // Initialize with "some" learning rate. The actual learning rate is set
    // later.
    let mut c_optimizer = tch::COptimizer::adamw(1e-5, beta1, beta2, wd, eps, amsgrad)
        .expect("Can't initialize AdamW optimizer");

    let vars = variables.lock().expect("Mutex poisoned");
    for var in &vars.trainable_variables {
        c_optimizer
            .add_parameters(&var.tensor, var.group)
            .expect("Can't add a trainable parameter to the optimizer");
    }

    c_optimizer
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check_epoch_config() {
        #[derive(Debug, serde::Deserialize)]
        struct LrConfig {
            epoch: EpochLearningRate,
        }

        let LrConfig { epoch } = toml::from_str(
            r#"
[epoch.linear-decrease]
base = 0.5
delta = 0.1
min = 0.3
"#,
        )
        .unwrap();
        assert!(matches!(epoch, EpochLearningRate::LinearDecrease { .. }));
        let LrConfig { epoch } = toml::from_str(
            r#"
[epoch.constant]
base = 0.5
"#,
        )
        .unwrap();
        assert!(matches!(epoch, EpochLearningRate::Constant { .. }));
    }

    #[test]
    fn check_scheduler_config() {
        #[derive(Debug, serde::Deserialize)]
        struct Cfg {
            inner: LearningRateScheduler,
        }

        let Cfg { inner } = toml::from_str(
            r#"
[inner.cosine-annealing]
restarts = 10
factor = 100
"#,
        )
        .unwrap();
        assert!(matches!(
            inner,
            LearningRateScheduler::CosineAnnealing { .. }
        ));

        let Cfg { inner } = toml::from_str(r#"[inner.constant]"#).unwrap();
        assert!(matches!(inner, LearningRateScheduler::Constant));
    }
}
