//! Dataset loader.

use camino::{Utf8Path, Utf8PathBuf};
use snafu::{ResultExt, Snafu};

use crate::TaggedMessage;

/// A dataset that combines multiple sources.
#[derive(
    Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct GlobalDataset {
    /// List of dataset sources.
    pub sources: Vec<Source>,
    /// Flat entries.
    pub entries: Vec<DataEntry>,
}

impl GlobalDataset {
    /// Returns the underlying tagged sentences.
    pub fn tagged_sentences(&self) -> impl ExactSizeIterator<Item = (&str, bool)> {
        self.entries.iter().map(|data_entry| {
            let spam = data_entry.entry.is_spam;
            let text = data_entry.entry.inner.text.as_str();
            (text, spam)
        })
    }

    /// Returns the amount of entries in the dataset.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the dataset contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Unable to dump a dataset.
#[derive(Debug, Snafu)]
pub enum DumpError {
    /// Unable to create a file.
    #[snafu(display("Unable to create a file at {path}"))]
    DumpCreateFile {
        /// Path to the file.
        path: Utf8PathBuf,
        /// Source error.
        source: std::io::Error,
    },
    /// Unable to serialize the dataset.
    #[snafu(display("Unable to serialize the dataset"))]
    #[snafu(context(false))]
    SerializeDump {
        /// Source error.
        source: serde_json::Error,
    },
}

#[derive(Debug, Snafu)]
pub enum LoadError {
    /// Unable to load a file.
    #[snafu(display("Unable to load the file at {path}"))]
    LoadFile {
        /// Path to the file.
        path: Utf8PathBuf,
        /// Source error.
        source: std::io::Error,
    },
    /// Can't deserialize a dataset.
    #[snafu(display("Can't deserialize a dataset"))]
    #[snafu(context(false))]
    DeserializeDump {
        /// Source error.
        source: serde_json::Error,
    },
}

impl GlobalDataset {
    /// Returns a source by its identifier.
    ///
    /// # Panics
    ///
    /// Will panic if source identifier is out of range.
    pub fn source(&self, source_id: SourceId) -> &Source {
        self.sources
            .get(source_id.0 as usize)
            .expect("Source identifier out of range")
    }

    /// Returns a path to the given source.
    ///
    /// # Panics
    ///
    /// Will panic if source identifier is out of range.
    pub fn source_path(&self, source_id: SourceId) -> &Utf8Path {
        &self.source(source_id).path
    }

    /// Adds a source to the data set and returns its identifier.
    pub fn add_source(&mut self, kind: SourceKind, path: Utf8PathBuf) -> SourceId {
        let id = SourceId(self.sources.len() as u64);
        self.sources.push(Source { kind, path });
        id
    }

    /// Dumps the dataset to the disk.
    pub fn dump<P>(&self, output: &P) -> Result<(), DumpError>
    where
        P: AsRef<Utf8Path>,
    {
        let path = output.as_ref();
        let output = std::io::BufWriter::new(
            std::fs::File::create(path).context(DumpCreateFileSnafu { path })?,
        );
        Ok(serde_json::to_writer(output, self)?)
    }

    /// Loads a dataset from a JSON file on the disk.
    pub fn load<P>(input: &P) -> Result<Self, LoadError>
    where
        P: AsRef<Utf8Path>,
    {
        let path = input.as_ref();
        let input = std::fs::read_to_string(path).context(LoadFileSnafu { path })?;
        Ok(serde_json::from_str(&input)?)
    }
}

/// Data source identifier.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SourceId(u64);

/// Position of a data entry in a data source.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SourcePosition(
    /// Position in the source file.
    pub u64,
);

/// A single data entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct DataEntry {
    /// A link to the source.
    pub source: SourceId,
    /// Position of the entry in the original source.
    pub source_position: SourcePosition,
    /// The actual entry.
    pub entry: TaggedMessage,
}

/// A description of a source of a dataset.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Source {
    /// Dataset kind.
    pub kind: SourceKind,
    /// Path to the original file.
    pub path: Utf8PathBuf,
}

impl Source {
    /// Displays the provided position.
    pub fn position(&self, source_position: SourcePosition) -> impl std::fmt::Display {
        struct Kek(i64);

        impl std::fmt::Display for Kek {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "id: {}", self.0)
            }
        }

        Kek(source_position.0 as i64)
    }
}

/// Data source type.
#[derive(
    Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    /// Exported telegram chat, but only forwards are parsed.
    TelegramForwards,
    /// Exported telegram chat (all messages are parsed).
    TelegramDump,
    /// Exported telegram "AI Guard" logs chat.
    TelegramAiGuardLogs,
}

impl std::fmt::Display for SourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SourceKind::TelegramForwards => "telegram export, forwards",
            SourceKind::TelegramDump => "telegram export",
            SourceKind::TelegramAiGuardLogs => "telegram export, AI Guard",
        };
        write!(f, "{s}")
    }
}
