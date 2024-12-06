use std::{num::NonZeroUsize, sync::Arc, time::Instant};

use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use nonzero_ext::nonzero;
use owo_colors::{AnsiColors, OwoColorize};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use snafu::ResultExt;
use tg_antispam::{
    bert::{BertConfig, BertModel},
    bert_classifier::BertClassifier,
    dataset::GlobalDataset,
    optimizer::{Optimizer, OptimizerConfiguration},
    tch::{self, nn::VarStore, Tensor},
    tokenizer::{BatchTokenizer, Tokenizer},
    training_progress::{EpochReportRecorder, ThresholdInfo},
    Freeze,
};

#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Spam classifier training.
#[derive(Parser)]
struct Args {
    /// Path to the train configuration TOML.
    #[arg(long = "config", default_value = "train.toml")]
    train_config: Utf8PathBuf,

    /// Size of the mini batch, i.e. the amount of sentences fed to the neural
    /// network at once.
    #[arg(long = "batch", default_value_t = 32)]
    batch_size: usize,

    /// Accumulate gradients for this amount of mini batches before updating the
    /// weights.
    #[arg(long = "macro-batch", default_value_t = 1)]
    macro_batch: usize,

    /// The amount of training epochs.
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Manual seed for torch.
    #[arg(long, default_value_t = 0xdeadf00d)]
    seed: i64,

    /// Output directory. Training logs and weights will go here.
    #[arg(long, default_value = "output")]
    output: Utf8PathBuf,
}

/// Training configuration.
#[derive(Debug, serde::Deserialize)]
pub struct TrainConfig {
    /// BERT-related configuration section.
    pub bert: TrainBertConfig,

    /// Learning rate configuration section.
    pub lr: OptimizerConfiguration,

    /// Training set.
    pub training_set: Utf8PathBuf,

    /// Development (validation) set.
    pub dev_set: Utf8PathBuf,

    /// Verification config.
    pub verification: VerificationConfig,

    /// Miscellaneous config.
    pub misc: MiscConfig,
}

/// Verification configuration.
#[derive(Debug, serde::Deserialize)]
pub struct VerificationConfig {
    /// Samples to verify each time.
    pub samples: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct TrainBertConfig {
    /// Path to the BERT configuration file.
    pub config: Utf8PathBuf,

    /// Path to the BERT's weights of the pre-trained model.
    pub weights: Utf8PathBuf,

    /// Path to the tokenizer.
    pub tokenizer: Utf8PathBuf,

    /// Path prefix of tensors name in the weights file.
    pub tensors_prefix: Option<String>,

    /// Whether to unfreeze the BERT layers.
    pub unfreeze_bert: bool,
}

/// Miscellaneous training parameters.
#[derive(Debug, serde::Deserialize)]
pub struct MiscConfig {
    /// When set, only this portion of the training set is selected on each
    /// epoch.
    ///
    /// The actual items are selected randomly each epoch.
    #[serde(default)]
    pub portion: Option<f32>,
}

fn main() -> snafu::Report<snafu::Whatever> {
    snafu::Report::capture(run)
}

const BERT_TENSORS_GROUP: NonZeroUsize = nonzero!(1usize);

fn run() -> Result<(), snafu::Whatever> {
    let Args {
        batch_size,
        macro_batch,
        epochs,
        train_config: train_config_path,
        output: output_dir,
        seed: torch_seed,
    } = Args::parse();

    // Set manual seeds for reproducibility.
    tch::manual_seed(torch_seed);
    // This seed might as well be configurable.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1337);

    // Create the output directory (and its parents).
    std::fs::create_dir_all(&output_dir).whatever_context("Can't create output directory")?;

    // Load the training configuration from the provided TOML file.
    let cfg: TrainConfig = {
        let train_config = std::fs::read_to_string(&train_config_path)
            .whatever_context("Can't read training configuration")?;

        toml::from_str(&train_config).whatever_context("Can't parse training configuration")?
    };
    println!("{cfg:#?}");

    // Copy configuration related files to the target (output) directory.
    std::fs::copy(&train_config_path, output_dir.join("train.toml"))
        .whatever_context("Can't copy configuration file to the output")?;
    std::fs::copy(&cfg.bert.config, output_dir.join("bert-config.json"))
        .whatever_context("Can't copy BERT configuration file to the output")?;
    // We need the tokenizer for the inference so copy it as well.
    std::fs::copy(&cfg.bert.tokenizer, output_dir.join("tokenizer.json"))
        .whatever_context("Can't copy BERT tokenizer file to the output")?;

    let TrainConfig {
        bert:
            TrainBertConfig {
                config: bert_config,
                weights: bert_weights,
                tokenizer: bert_tokenizer,
                tensors_prefix: bert_tensors_prefix,
                unfreeze_bert,
            },
        lr,
        training_set,
        dev_set,
        verification: VerificationConfig {
            samples: verification_samples,
        },
        misc: MiscConfig {
            portion: training_portion,
        },
    } = cfg;

    // Normally this is a configurable parameter.
    let device = tch::Device::Cuda(0);
    // This is a variables (tensors) storage.
    let mut var_store = VarStore::new(device);

    // BERT configuration is just a JSON file.
    let bert_config: BertConfig = {
        let bert_config = std::fs::read_to_string(bert_config)
            .whatever_context("Can't read BERT configuration")?;

        serde_json::from_str(&bert_config).whatever_context("Can't parse BERT configuration")?
    };

    // Loads the tokenizer from the `tokenizer.json`.
    let tokenizer =
        Tokenizer::from_json(bert_tokenizer, bert_config.max_position_embeddings as usize)
            .whatever_context("Load tokenizer")?;
    // We are sharing the tokenizer across the threads so wrap it up in the
    // `Arc`.
    let tokenizer = Arc::new(tokenizer);

    // Every variable in the `VarStore` is stored at a certain path. Depending
    // on the configuration this path may vary.
    let bert_tensors_path = if let Some(bert_tensors_prefix) = &bert_tensors_prefix {
        var_store.root() / bert_tensors_prefix
    } else {
        var_store.root()
    };

    // Prepare the pretrained model.
    let bert_model = BertModel::new_with_optional_pooler(
        bert_tensors_path.set_group(BERT_TENSORS_GROUP.get()),
        &bert_config,
        false,
    );

    // Load the pre-trained weights. It is super IMPORTANT that only those
    // variables that are registered so far are loaded! And it is also super
    // IMPORTANT to call the `load` AFTER the required variables are registered.
    // All the variables that are registered later will not be loaded from the
    // disk.
    var_store
        .load(bert_weights)
        .whatever_context("Can't load BERT weights")?;

    // Some variable names manipulations.
    if let Some(bert_tensors_prefix) = bert_tensors_prefix {
        if bert_tensors_prefix != "bert" {
            // Rename required.
            let old_prefix = {
                let mut prefix = bert_tensors_prefix;
                prefix.push('.');
                prefix
            };
            let mut vars = var_store.variables_.lock().expect("Mutex poisoned");
            let named_variables = vars
                .named_variables
                .drain()
                .map(|(key, value)| {
                    let raw_name = key.strip_prefix(&old_prefix).expect("Must be there");
                    let key = format!("bert.{raw_name}");
                    (key, value)
                })
                .collect();
            vars.named_variables = named_variables;
        } else {
            // The prefix is already correct.
        }
    } else {
        // Add "bert." prefix to all the tensors.
        let mut vars = var_store.variables_.lock().expect("Mutex poisoned");
        let named_variables = vars
            .named_variables
            .drain()
            .map(|(key, value)| {
                let key = format!("bert.{key}");
                (key, value)
            })
            .collect();
        vars.named_variables = named_variables;
    }

    println!("BERT loaded");

    // Load the "dev" dataset. This dataset is used to validate how good the
    // network performs.
    let dev_dataset = GlobalDataset::load(&dev_set).whatever_context("Loading dev set")?;
    snafu::ensure_whatever!(!dev_dataset.entries.is_empty(), "Dev input is empty");

    // Load the "train" dataset. This dataset is used to actually train the
    // model.
    let training_dataset =
        GlobalDataset::load(&training_set).whatever_context("Loading training set")?;
    snafu::ensure_whatever!(
        !training_dataset.entries.is_empty(),
        "Training input is empty"
    );

    // Count how many spam and ham (not spam) messages we've got in the training
    // dataset.
    let mut spam_count = 0usize;
    let mut ham_count = 0usize;
    for entry in &training_dataset.entries {
        if entry.entry.is_spam {
            spam_count += 1;
        } else {
            ham_count += 1;
        }
    }
    println!(
        "Training set: {ham_count} normal messages and {spam_count} spam messages, \
         ham to spam ratio: {:.3}",
        ham_count as f32 / spam_count as f32
    );

    println!("Input loaded");

    // Initialize the classifier. Its weights (apart from the pre-trained BERT
    // weights) are initialized randomly.
    let bert_classifier = BertClassifier::new(
        bert_model,
        bert_config.hidden_size,
        &var_store.root(),
        tokenizer.pad_token(),
    );

    // Optionally "unfreeze" the BERT layers: we might choose whether we only
    // want to train the "head" (a layer on top of the pre-trained BERT), or the
    // full network (the layer on top + the pre-trained BERT).
    if unfreeze_bert {
        bert_classifier.bert().unfreeze();
        println!("Unfrozen all BERT weights");
    } else {
        bert_classifier.bert().freeze();
        println!("Frozen all BERT weights");
    }

    // Tokenization is fast, so tokenize each sentence in advance.
    let mut tokenized_training = {
        let begin = Instant::now();
        let mut tokenized = Vec::new();
        let mut batched =
            BatchTokenizer::new(&tokenizer, batch_size, training_dataset.tagged_sentences());
        while let Some(batch) = batched
            .process_parallel()
            .whatever_context("Training data set tokenization")?
        {
            tokenized.extend(batch.iter().cloned());
        }
        let elapsed = begin.elapsed();
        println!(
            "Training set ({} entries) tokenized in {elapsed:?}",
            training_dataset.len()
        );
        tokenized
    };

    // Tokenize the sentences from the "dev" dataset.
    let dev_data = {
        let mut dev_data = Vec::new();
        let mut batched =
            BatchTokenizer::new(&tokenizer, batch_size, dev_dataset.tagged_sentences());
        while let Some(batch) = batched
            .process_parallel()
            .whatever_context("Dev data set tokenization error")?
        {
            dev_data.extend(batch);
        }
        dev_data
    };

    // We may only want a portion of the data.
    let training_items = training_portion
        .map(|portion| {
            let len = tokenized_training.len() as f32 * portion.clamp(0., 1.);
            let len = len.round() as usize;
            len.min(tokenized_training.len())
        })
        .unwrap_or_else(|| tokenized_training.len());

    // Pre-calculate the amount of steps each epoch will take.
    let total_steps = tokenized_training[..training_items]
        .chunks(batch_size)
        .len();

    // Initialize the "optimizer". This tool guides the "learning rate"
    // parameter throughout every epoch.
    let mut optimizer = Optimizer::new(
        &var_store,
        lr,
        total_steps,
        unfreeze_bert.then_some(BERT_TENSORS_GROUP),
        macro_batch,
    );

    let pos_weight = {
        let pos_weight: f64 = ham_count as f64 / spam_count as f64;
        println!("Positive samples weight set to {pos_weight}");
        Tensor::scalar_tensor(pos_weight, (tch::Kind::Float, device))
    };

    let mut best_result = 0.;
    for epoch in 0..epochs {
        let mut epoch_optimizer = optimizer.start_epoch();
        println!(
            "Epoch #{epoch}, base learning rate = {:.3e}",
            epoch_optimizer.base_learning_rate()
        );

        let mut loss_reporter = LossReporter::new(500, batch_size, training_items);
        let mut progress_info = EpochReportRecorder::new(epoch, training_items);

        // Do not tokenize the sentences for each epoch, instead reuse the
        // tokenization results while randomly shuffling the order in each
        // epoch.
        tokenized_training.shuffle(&mut rng);

        let begin = Instant::now();
        for batch in tokenized_training[..training_items].chunks(batch_size) {
            // We push the tokenized sentences through the network and calculate
            // how the predicted results differs from the expected.
            let loss_tensor = bert_classifier
                .train(
                    batch
                        .iter()
                        .map(|(tokens, is_spam)| (tokens.as_slice(), *is_spam)),
                    Some(&pos_weight),
                )
                .whatever_context("Train failed")?;
            let loss = f32::try_from(&loss_tensor).expect("Must be a float");
            progress_info.record(batch.len(), epoch_optimizer.learning_rate(), loss);
            loss_reporter.report(loss);
            epoch_optimizer.next_mini_batch(&loss_tensor);
        }
        let elapsed = begin.elapsed();
        epoch_optimizer.finish();
        drop(loss_reporter);

        // Since the model outputs the probability of a message being a spam, we
        // need to know a threshold level at which we decide a message to
        // actually be a spam.
        let histograms = asses_model(&dev_data, batch_size, &bert_classifier)?;
        // So we calculate the threshold so that it maximizes the f0.5 score on
        // the dev dataset.
        let (best_threshold, f_score) = histograms.best(0.5);

        let color = if f_score >= best_result {
            AnsiColors::Green
        } else {
            AnsiColors::Cyan
        };

        println!(
            "Epoch #{epoch} trained in {:?}, F0.5 {:.3}%, avg loss = {}, best threshold = {:.2}",
            elapsed,
            (f_score * 100.).color(color),
            progress_info.avg_loss(),
            best_threshold as f32 / 100.,
        );
        let model_path = output_dir.join("last.safetensors");
        save_model(&var_store, &model_path)?;

        if f_score >= best_result {
            let previous_best = std::mem::replace(&mut best_result, f_score);
            best_result = f_score;
            println!(
                "Epoch #{epoch}. {}: {:.3}%, previous was {:.3}%",
                "This is the best result so far".green(),
                (f_score * 100.).green(),
                (previous_best * 100.).cyan()
            );
            let best = output_dir.join("best.safetensors");
            std::fs::copy(model_path, &best).whatever_context("Can't copy the best model")?;
            println!("Epoch #{epoch}. Saved best result to {best}");
        } else {
            println!(
                "Epoch #{epoch}. Best result so far: {:.3}%",
                (best_result * 100.).cyan()
            );
        }

        let verification = predict_samples(&verification_samples, &bert_classifier, &tokenizer);
        let report = progress_info.finalize(&verification_samples, verification, histograms);
        report
            .save(output_dir.join(format!("epoch_{epoch:0>4}.txt")))
            .whatever_context("Can't save report")?;
        report
            .plot(output_dir.join(format!("epoch_{epoch:0>4}.png")))
            .whatever_context("Can't save report plot")?;
    }

    Ok(())
}

/// A helper function to save the calculated weights to a file.
fn save_model<P>(vars: &VarStore, output: P) -> Result<(), snafu::Whatever>
where
    P: AsRef<Utf8Path>,
{
    let output = output.as_ref();
    let directory = output.parent().unwrap_or(Utf8Path::new("."));

    let temp = tempfile::Builder::new()
        .suffix(".safetensors")
        .tempfile_in(directory)
        .whatever_context("Can't create temp file")?;
    vars.save(temp.path())
        .whatever_context("Can't save weights")?;
    temp.persist(output)
        .whatever_context("Can't persist the output")?;

    println!("Saved tensors to {output}");

    Ok(())
}

/// Predicts a probability of being a spam for each of the sentences.
fn predict_samples<I>(sentences: I, pos_tagger: &BertClassifier, tokenizer: &Tokenizer) -> Tensor
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let _guard = tch::no_grad_guard();
    let output = pos_tagger
        .predict(sentences.into_iter().map(|sentence| {
            let sentence = sentence.as_ref();
            tokenizer.tokenize_sentence(sentence).expect("Must be OK")
        }))
        .expect("Prediction failed");
    output
}

/// Assesses how well the trained model performs on the provided "dev" dataset
/// for each threshold level.
fn asses_model(
    dev_data: &[(Vec<i64>, bool)],
    batch_size: usize,
    pos_tagger: &BertClassifier,
) -> Result<ThresholdInfo, snafu::Whatever> {
    let _guard = tch::no_grad_guard();
    let begin = Instant::now();

    let total_elements = dev_data.len();
    let bar = indicatif::ProgressBar::new(total_elements as u64).with_style(
        indicatif::ProgressStyle::with_template(
            "[{elapsed:>4}/{duration:<4}] {bar:40.cyan/blue} {pos:>7}/{len:7}",
        )
        .expect("Must be OK"),
    );

    let mut threshold_info = ThresholdInfo::default();

    for batch in dev_data.chunks(batch_size) {
        let output = pos_tagger
            .predict(batch.iter().map(|(tokens, _is_spam)| tokens))
            .whatever_context("Prediction failed")?;

        let predictions = output.split(1, 0);

        for ((_tokens, is_spam), prediction) in batch.iter().zip(&predictions) {
            let predicted_spam: f32 = prediction.squeeze_dim(0).try_into().expect("Must be OK");
            threshold_info.record_prediction(*is_spam, predicted_spam);
        }

        bar.inc(batch.len() as u64);
    }
    bar.finish_and_clear();
    println!("assessment performed in {:?}", begin.elapsed());

    Ok(threshold_info)
}

/// A helper type to report the running "loss".
struct LossReporter {
    loss_sum: f32,
    loss_counter: usize,
    report_interval: usize,
    batch_size: usize,

    bar: indicatif::ProgressBar,
}

impl LossReporter {
    pub fn new(report_interval: usize, batch_size: usize, total_elements: usize) -> Self {
        let bar = indicatif::ProgressBar::new(total_elements as u64).with_style(
            indicatif::ProgressStyle::with_template(
                "[{elapsed:>4}/{duration:<4}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .expect("Must be OK"),
        );
        Self {
            loss_sum: 0.,
            loss_counter: 0,
            report_interval: report_interval.max(1),
            batch_size,
            bar,
        }
    }

    pub fn report(&mut self, loss: f32) {
        if self.loss_counter * self.batch_size >= self.report_interval {
            let avg_loss = self.loss_sum / self.loss_counter as f32;
            self.bar.inc((self.loss_counter * self.batch_size) as u64);
            self.bar.set_message(format!("Average loss: {avg_loss}"));
            self.loss_sum = 0.;
            self.loss_counter = 0;
        }
        self.loss_sum += loss;
        self.loss_counter += 1;
    }
}

impl Drop for LossReporter {
    fn drop(&mut self) {
        self.bar.finish_and_clear();
    }
}
