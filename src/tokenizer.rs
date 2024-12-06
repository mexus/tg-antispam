//! A convenient wrapper around a 🤗 tokenizer with a nicer API.

use camino::Utf8Path;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use snafu::{OptionExt, ResultExt, Snafu};

/// Sentence tokenization.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Words tokenizer.
    inner: tokenizers::Tokenizer,
    /// Maximum tokens per sentence.
    max_tokens: usize,

    /// `[CLS]` token.
    cls_token: i64,
    /// `[SEP]` token.
    sep_token: i64,
    /// `[PAD]` token.
    pad_token: i64,
    /// `[MASK]` token.
    mask_token: i64,
}

/// Failed to tokenize a sentence.
#[derive(Debug, Snafu)]
#[snafu(display("Failed to tokenize a sentence"))]
#[snafu(context(false))]
pub struct TokenizationError {
    source: Box<dyn std::error::Error + Send + Sync>,
}

/// Unable to initialize a tokenizer.
#[derive(Debug, Snafu)]
pub enum InitTokenizerError {
    /// Failed to read the tokenizer.
    #[snafu(display("Failed to read the tokenizer"))]
    #[snafu(context(false))]
    ReadFile {
        /// Source error.
        source: std::io::Error,
    },

    /// Failed to load a vocabulary.
    #[snafu(display("Failed to load a tokenizer"))]
    #[snafu(context(false))]
    LoadTokenizer {
        /// Source error.
        source: serde_json::Error,
    },

    /// A token is missing.
    #[snafu(display("The token {token:?} is missing from the vocabulary"))]
    MissingToken {
        /// The missing token/
        token: &'static str,
    },
}

impl Tokenizer {
    /// Builds the tokenizer from the serialized JSON at the provided path.
    pub fn from_json<P>(path: P, max_tokens: usize) -> Result<Self, InitTokenizerError>
    where
        P: AsRef<Utf8Path>,
    {
        let path = path.as_ref();

        let data = std::fs::read_to_string(path)?;

        let inner: tokenizers::Tokenizer = serde_json::from_str(&data)?;

        let cls_token = inner
            .token_to_id("[CLS]")
            .context(MissingTokenSnafu { token: "[CLS]" })?;
        let sep_token = inner
            .token_to_id("[SEP]")
            .context(MissingTokenSnafu { token: "[SEP]" })?;
        let pad_token = inner
            .token_to_id("[PAD]")
            .context(MissingTokenSnafu { token: "[PAD]" })?;
        let mask_token = inner
            .token_to_id("[MASK]")
            .context(MissingTokenSnafu { token: "[MASK]" })?;

        Ok(Self {
            inner,
            max_tokens,
            cls_token: i64::from(cls_token),
            sep_token: i64::from(sep_token),
            pad_token: i64::from(pad_token),
            mask_token: i64::from(mask_token),
        })
    }

    /// Tokenizes the provided sentence.
    pub fn tokenize_sentence<S>(&self, sentence: S) -> Result<Vec<i64>, TokenizationError>
    where
        S: AsRef<str>,
    {
        let sentence = sentence.as_ref();
        let mut tokens = vec![self.cls_token];
        let encoded = self.inner.encode(sentence, false)?;

        tokens.extend(
            encoded
                .get_ids()
                .iter()
                .copied()
                .map(i64::from)
                .take(self.max_tokens - 2),
        );
        tokens.push(self.sep_token);
        Ok(tokens)
    }

    /// Returns the `[PAD]` token ID.
    pub fn pad_token(&self) -> i64 {
        self.pad_token
    }

    /// Returns the `[MASK]` token ID.
    pub fn mask_token(&self) -> i64 {
        self.mask_token
    }

    /// Gets a token for the value.
    pub fn get_token(&self, value: &str) -> Option<i64> {
        self.inner.token_to_id(value).map(i64::from)
    }
}

type Sentence = (Vec<i64>, bool);

/// Tokenizes multiple sentences at once in parallel.
pub struct BatchTokenizer<'a, Sentences> {
    tokenizer: &'a Tokenizer,
    batch_size: usize,
    incoming: std::iter::Enumerate<Sentences>,
    encoded: Vec<(usize, Result<Sentence, TokenizationError>)>,
}

impl<'a, Sentences> BatchTokenizer<'a, Sentences> {
    /// Construct a batch tokenizer.
    pub fn new<I>(
        tokenizer: &'a Tokenizer,
        batch_size: usize,
        incoming: I,
    ) -> BatchTokenizer<Sentences>
    where
        I: IntoIterator<IntoIter = Sentences>,
        Sentences: Iterator,
    {
        BatchTokenizer {
            tokenizer,
            batch_size,
            incoming: incoming.into_iter().enumerate(),
            encoded: Vec::with_capacity(batch_size),
        }
    }
}

/// Batch encoding error.
#[derive(Debug, Snafu)]
pub enum BatchError {
    /// Unable to encode a sentence.
    #[snafu(display("Unable to encode the sentence #{sentence_number}"))]
    EncodeSentence {
        /// Sentence number.
        sentence_number: usize,
        /// Source error.
        source: TokenizationError,
    },
}

impl<'a, Sentences, W> BatchTokenizer<'a, Sentences>
where
    Sentences: Iterator<Item = (W, bool)>,
    W: AsRef<str>,
{
    /// Processes a batch.
    pub fn process_parallel(&mut self) -> Result<Option<Vec<Sentence>>, BatchError>
    where
        <Sentences as Iterator>::Item: Send,
    {
        let batch = self
            .incoming
            .by_ref()
            .take(self.batch_size)
            .collect::<Vec<_>>();
        if batch.is_empty() {
            return Ok(None);
        }

        batch
            .into_par_iter()
            .map(|(sentence_number, (sentence, class))| {
                (
                    sentence_number,
                    self.tokenizer
                        .tokenize_sentence(sentence)
                        .map(|tokens| (tokens, class)),
                )
            })
            .collect_into_vec(&mut self.encoded);
        Ok(Some(
            self.encoded
                .drain(..)
                .map(|(sentence_number, encoded)| {
                    encoded.context(EncodeSentenceSnafu { sentence_number })
                })
                .collect::<Result<_, _>>()?,
        ))
    }
}
