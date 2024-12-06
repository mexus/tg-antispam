use std::io::BufRead;

use camino::Utf8PathBuf;
use clap::Parser;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::Session,
    tensor::TensorElementType,
};
use snafu::{OptionExt, ResultExt};
use tg_antispam::tokenizer::Tokenizer;

#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Spam classifier inference (ONNX mode).
#[derive(Parser)]
struct Args {
    /// Path to the BERT configuration json.
    #[arg(long, default_value = "trained/config.json", env = "BERT_CONFIG")]
    bert_config: Utf8PathBuf,

    /// Path to the BERT tokenizer.
    #[arg(long, default_value = "trained/tokenizer.json", env = "BERT_TOKENIZER")]
    bert_tokenizer: Utf8PathBuf,

    /// ONNX file.
    #[arg(short, long, default_value = "my.onnx", env = "MODEL_WEIGHTS")]
    weights: Utf8PathBuf,

    /// When enabled runs on CPU. Otherwise runs on CUDA.
    #[arg(long)]
    cpu: bool,

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
        file: sentence_file,
        sentence: mut sentences,
        cpu,
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

    let ort_provider = if cpu {
        CPUExecutionProvider::default().build().error_on_failure()
    } else {
        CUDAExecutionProvider::default().build().error_on_failure()
    };
    let session = Session::builder()
        .whatever_context("session builder")?
        .with_execution_providers([ort_provider])
        .whatever_context("set execution providers")?
        .commit_from_file(weights)
        .whatever_context("Commit model file")?;

    let half = match session.outputs[0]
        .output_type
        .tensor_type()
        .whatever_context("Output is not a tensor?")?
    {
        TensorElementType::Float32 => false,
        TensorElementType::Float16 | TensorElementType::Bfloat16 => true,
        type_ => snafu::whatever!("Unexpected tensor type {type_:?}"),
    };

    let tokenized_sentences = sentences
        .iter()
        .map(|sentence| {
            // Normally tokenization could be performed in a separate
            // thread, but since in this module we run everything on CPU
            // that won't make a lot of sense.
            tokenizer
                .tokenize_sentence(sentence)
                .whatever_context("Can't tokenize a sentence")
        })
        .collect::<Result<Vec<_>, _>>()?;

    let max_length = tokenized_sentences
        .iter()
        .map(Vec::len)
        .max()
        .expect("Not empty");

    let mut tokens = ndarray::Array2::from_elem((tokenized_sentences.len(), max_length), pad_token);
    let mut attention = ndarray::Array2::zeros((tokenized_sentences.len(), max_length));

    // Fill the `tokens` and `attention` tensors.
    for ((sentence, tokens), attention) in tokenized_sentences
        .iter()
        .zip(tokens.rows_mut())
        .zip(attention.rows_mut())
    {
        sentence
            .iter()
            .zip(tokens)
            .zip(attention)
            .for_each(|((token, t), att)| {
                *t = *token;
                *att = 1i64;
            });
    }

    let output = session
        .run(
            ort::inputs![
                "input_ids" => tokens,
                "attention_mask" => attention,
            ]
            .expect("Inputs must be fine"),
        )
        .whatever_context("onnx failure")?;
    let output = output.get("output").expect("Must be there");

    let output: ndarray::ArrayD<f32> = if half {
        output
            .try_extract_tensor::<half::f16>()
            .expect("Must be fine")
            .squeeze()
            .map(|x| x.to_f32())
    } else {
        output
            .try_extract_tensor::<f32>()
            .expect("Must be fine")
            .squeeze()
            .to_owned()
    };

    for (sentence, spam_logit) in sentences.iter().zip(output) {
        let spam_probability = sigmoid(spam_logit);
        println!("spam {:5.2}%: {sentence}", spam_probability * 100.);
    }

    Ok(())
}

fn sigmoid(logit: f32) -> f32 {
    let exp = logit.exp();
    exp / (1. + exp)
}
