use std::io::BufRead;

use camino::Utf8PathBuf;
use clap::Parser;
use snafu::ResultExt;
use tch::nn::VarStore;
use tg_antispam::{bert_classifier::BertClassifier, tch, tokenizer::Tokenizer};

#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Spam classifier inference.
#[derive(Parser)]
struct Args {
    /// Path to the BERT configuration json.
    #[arg(long, default_value = "trained/config.json", env = "BERT_CONFIG")]
    bert_config: Utf8PathBuf,

    /// Path to the BERT tokenizer.
    #[arg(long, default_value = "trained/tokenizer.json", env = "BERT_TOKENIZER")]
    bert_tokenizer: Utf8PathBuf,

    /// Weights file.
    #[arg(
        short,
        long,
        default_value = "trained/model.safetensors",
        env = "MODEL_WEIGHTS"
    )]
    weights: Utf8PathBuf,

    /// File with line-separated sentences.
    #[arg(long)]
    file: Option<Utf8PathBuf>,

    /// Sentence to parse.
    sentence: Vec<String>,
}

fn main() -> snafu::Report<snafu::Whatever> {
    snafu::Report::capture(run)
}

fn run() -> Result<(), snafu::Whatever> {
    let Args {
        bert_config,
        bert_tokenizer,
        weights,
        sentence: mut sentences,
        file: sentence_file,
    } = Args::parse();

    if let Some(sentence_file) = sentence_file {
        // Read sentences from the file line-by-line.
        let mut reader = std::io::BufReader::new(
            std::fs::File::open(sentence_file)
                .whatever_context("Can't open the file with sentences")?,
        );
        let mut line = String::new();
        while {
            line.clear();
            reader
                .read_line(&mut line)
                .whatever_context("Can't read a sentence")?
        } != 0
        {
            let line = line.trim();
            let line = line.replace(['«', '»'], "\"");
            sentences.push(line);
        }
    }

    // Nothing to infer.
    snafu::ensure_whatever!(!sentences.is_empty(), "No sentences provided");

    // Normally this is a configurable parameter.
    let device = tch::Device::Cpu;
    // This is a variables (tensors) storage.
    let mut var_store = VarStore::new(device);

    // BERT configuration is just a JSON file.
    let bert_config: tg_antispam::bert::BertConfig = {
        let bert_config = std::fs::read_to_string(bert_config)
            .whatever_context("Can't read BERT configuration")?;

        serde_json::from_str(&bert_config).whatever_context("Can't parse BERT configuration")?
    };

    // Loads the tokenizer from the `tokenizer.json`.
    let tokenizer =
        Tokenizer::from_json(bert_tokenizer, bert_config.max_position_embeddings as usize)
            .whatever_context("Can't load BERT tokenizer")?;
    // We need to know ID of the "padding" in order to pad the sentences which
    // are not of equal length.
    let pad_token = tokenizer.pad_token();

    // Initializes the tagger.
    let pos_tagger = BertClassifier::inference(&bert_config, &var_store.root(), pad_token);

    // Finally load the weights. It is super IMPORTANT that this step must be
    // performed AFTER all the variables are registered within the `VarStore`!
    var_store
        .load(weights)
        .whatever_context("Can't load weights")?;

    // Calculate the output of the model for all the sentence simultaneously (in
    // batch).
    let output = pos_tagger
        .predict(
            sentences
                .iter()
                .map(|sentence| {
                    // Normally tokenization could be performed in a separate
                    // thread, but since in this module we run everything on CPU
                    // that won't make a lot of sense.
                    tokenizer
                        .tokenize_sentence(sentence)
                        .whatever_context("Can't tokenize a sentence")
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .whatever_context("Prediction failed")?;
    for (id, (sentence, predicted)) in sentences.into_iter().zip(output.split(1, 0)).enumerate() {
        // Our model outputs a single float for each sentence: the probability
        // of it being a spam.
        let probability: f32 = predicted.try_into().expect("Must be a float");
        let probability = probability * 100.;
        println!("\n#{id}. spam: {probability:.2}%\n#{id} sentence: {sentence}");
    }

    Ok(())
}
