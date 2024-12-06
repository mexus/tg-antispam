//! A neural BERT-based spam detection.

pub mod bert_classifier;
pub mod dataset;
pub mod freeze;
pub mod optimizer;
pub mod tokenizer;
pub mod training_progress;

pub use freeze::Freeze;

/// Re-export tch-rs.
pub use tch;

// The following modules are copied from the rust-bert project for the purpose
// of accessing the transformer layers.
mod activations;
mod attention;
pub mod bert;
mod dropout;

/// A telegram message after the preprocessing step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct PreprocessedMessage {
    /// The text.
    pub text: String,

    /// Who sent the message (name).
    pub from: String,

    /// Who sent the message (ID).
    pub from_id: String,

    /// Where the message was found.
    pub chat_id: i64,

    /// When the message was sent.
    #[serde(with = "time::serde::timestamp")]
    pub time: time::OffsetDateTime,

    /// Whom the messages was forwarded from.
    #[serde(
        default,
        deserialize_with = "deserialize_some",
        skip_serializing_if = "Option::is_none"
    )]
    pub forwarded_from: Option<Option<String>>,
}

/// Any value that is present is considered Some value, including null.
fn deserialize_some<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: serde::Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    serde::Deserialize::deserialize(deserializer).map(Some)
}

/// A tagged message.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaggedMessage {
    /// The message itself.
    #[serde(flatten)]
    pub inner: PreprocessedMessage,
    /// Whether the message is a spam or not.
    pub is_spam: bool,
}
