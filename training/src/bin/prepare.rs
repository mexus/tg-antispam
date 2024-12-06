//! Prepares data for training.

use std::{collections::HashMap, time::Instant};

use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use itertools::Itertools;
use rand::{distributions::Bernoulli, Rng, SeedableRng};
use regex::Regex;
use snafu::{OptionExt, ResultExt};
use tg_antispam::dataset::{DataEntry, GlobalDataset, SourceKind};
use training::{preprocess_message, TelegramExportData, TgEntityType, TgMessage};

#[snafu::report]
fn main() -> Result<(), snafu::Whatever> {
    run()
}

/// Prepares data for the training.
#[derive(Debug, Parser)]
struct Args {
    /// Path to the configuration file.
    #[arg(long, short, default_value = "prepare.toml")]
    config: Utf8PathBuf,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Config {
    /// Data sources.
    sources: HashMap<String, Source>,
    /// Which portion of data to split to the dev set. It is only applicable for
    /// the data source that go both to the train and the dev data sets.
    dev_portion: f64,
    /// Path where train file will go.
    train_output: Utf8PathBuf,
    /// Path where dev file will go.
    dev_output: Utf8PathBuf,
    /// Initial random seed.
    #[serde(default = "defaults::default_seed")]
    seed: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Source {
    /// Path or paths to the data source file(s) or directory(ies) containing it
    /// (them).
    #[serde(deserialize_with = "parse::parse_one_or_many")]
    #[serde(alias = "paths")]
    path: Vec<Utf8PathBuf>,

    /// If set, this regular expression is applied to keep only matching paths.
    /// Only file names are matched!
    #[serde(deserialize_with = "parse::parse_regex", default = "defaults::fn_none")]
    filename_filter: Option<Regex>,

    /// Spam marker. All the entries in the file are considered to be a spam
    /// according to this marker.
    is_spam: bool,

    /// Only this (random) part of the file is taken.
    #[serde(default = "defaults::default_source_part")]
    partial: f64,

    /// Whether this source should be included in the training dataset.
    #[serde(default = "defaults::fn_true")]
    train: bool,

    /// Whether this source should be included in the "dev" dataset.
    #[serde(default = "defaults::fn_true")]
    dev: bool,

    /// Kind of the source file.
    kind: SourceKind,
}

mod defaults {
    pub const fn default_seed() -> u64 {
        1337
    }

    pub const fn fn_none<T>() -> Option<T> {
        None
    }

    pub const fn fn_true() -> bool {
        true
    }

    pub const fn default_source_part() -> f64 {
        1.
    }
}

mod parse {
    use regex::Regex;
    use serde::{Deserialize, Deserializer};

    pub fn parse_regex<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Regex>, D::Error> {
        let regex = Option::<String>::deserialize(d)?;
        regex
            .as_deref()
            .map(Regex::new)
            .transpose()
            .map_err(<D::Error as serde::de::Error>::custom)
    }

    pub fn parse_one_or_many<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum OneOrMany<T> {
            One(T),
            Many(Vec<T>),
        }
        Ok(match OneOrMany::deserialize(d)? {
            OneOrMany::One(one) => vec![one],
            OneOrMany::Many(many) => many,
        })
    }
}

fn run() -> Result<(), snafu::Whatever> {
    let Args { config } = Args::parse();
    let config = std::fs::read_to_string(&config).whatever_context("Can't read config file")?;
    let config: Config = toml::from_str(&config).whatever_context("Can't parse config file")?;

    let mut train_dataset = GlobalDataset::default();
    let mut dev_dataset = GlobalDataset::default();

    let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(config.seed);

    for (source_number, (source_name, source)) in config.sources.into_iter().enumerate() {
        let kind = source.kind;
        let mut all_files = vec![];
        for path in source.path {
            if path.is_dir() {
                // Traverse the directory recursively.
                let paths = walkdir::WalkDir::new(&path).into_iter().filter_entry(|e| {
                    if !e.path().is_file() {
                        false
                    } else if let Some(filter) = &source.filename_filter {
                        Utf8Path::from_path(e.path())
                            .and_then(Utf8Path::file_name)
                            .map(|file_name| filter.is_match(file_name))
                            .unwrap_or(false)
                    } else {
                        true
                    }
                });
                for path in paths {
                    let entry = path.whatever_context("Can't fetch a directory entry")?;
                    let all_paths = Utf8Path::from_path(entry.path())
                        .whatever_context("Non-unicode file path")?
                        .to_owned();
                    all_files.push(all_paths);
                }
            } else {
                all_files.push(path);
            }
        }

        let include_distribution =
            Bernoulli::new(source.partial).whatever_context("Incorrect 'part' value")?;

        let dev_distribution_p = match (source.train, source.dev) {
            (true, true) => config.dev_portion,
            (true, false) => 0.,
            (false, true) => 1.,
            (false, false) => {
                snafu::whatever!(
                    "The data source {source_name} (#{source_number}) goes nowhere :)"
                );
            }
        };
        let dev_distribution =
            Bernoulli::new(dev_distribution_p).whatever_context("Invalid dev probability")?;

        for path in all_files {
            let source_id = train_dataset.add_source(kind, path.clone());
            let dev_source_id = dev_dataset.add_source(kind, path.clone());
            assert_eq!(source_id, dev_source_id);

            let mut loaded_counter = 0usize;
            let mut processed_counter = 0usize;
            let begin = Instant::now();
            match kind {
                SourceKind::TelegramForwards
                | SourceKind::TelegramDump
                | SourceKind::TelegramAiGuardLogs => {
                    load_tg_export(&path, source_id, source.is_spam, kind)
                        .with_whatever_context(|_| format!("Processing file {path}"))?
                        .filter(|entry| !entry.entry.inner.text.is_empty())
                        .for_each(|entry| {
                            loaded_counter += 1;
                            if !rng.sample(include_distribution) {
                                return;
                            }
                            if rng.sample(dev_distribution) {
                                dev_dataset.entries.push(entry);
                            } else {
                                train_dataset.entries.push(entry);
                            }
                            processed_counter += 1;
                        });
                }
            }
            let elapsed = begin.elapsed();
            eprintln!(
                "{path}: \
                 loaded {loaded_counter} and \
                 processed {processed_counter} \
                 messages in {elapsed:?}"
            );
        }
    }

    dev_dataset
        .dump(&config.dev_output)
        .whatever_context("Can't dump dev dataset")?;
    train_dataset
        .dump(&config.train_output)
        .whatever_context("Can't dump train dataset")?;

    Ok(())
}

/// Parses telegram JSON export chat file.
fn load_tg_export(
    path: &Utf8Path,
    source_id: tg_antispam::dataset::SourceId,
    is_spam: bool,
    kind: SourceKind,
) -> Result<impl Iterator<Item = DataEntry>, snafu::Whatever> {
    let begin = Instant::now();
    let file_data = std::fs::read_to_string(path).whatever_context("Can't read data source")?;
    let TelegramExportData {
        id: chat_id,
        messages,
    } = serde_json::from_str(&file_data).whatever_context("Can't parse data source")?;
    let elapsed = begin.elapsed();
    eprintln!("Loaded {path} in {elapsed:?}");

    let messages = messages.into_iter();
    let iter = match kind {
        SourceKind::TelegramForwards | SourceKind::TelegramDump => itertools::Either::Left(
            messages
                .filter(move |message| {
                    !matches!(kind, SourceKind::TelegramForwards)
                        || message.forwarded_from.is_some()
                })
                .filter_map(move |message| {
                    let message_id = message.id as u64;
                    Some((message_id, preprocess_message(message, chat_id)?))
                }),
        ),
        SourceKind::TelegramAiGuardLogs => itertools::Either::Right(
            filter_ai_guard_logs(messages, chat_id).filter_map(|result| match result {
                Ok(x) => x,
                Err(err) => {
                    panic!("{}", snafu::Report::from_error(err))
                }
            }),
        ),
    };

    let iter = iter.map(
        move |(message_id, preprocessed)| tg_antispam::dataset::DataEntry {
            source: source_id,
            source_position: tg_antispam::dataset::SourcePosition(message_id),
            entry: tg_antispam::TaggedMessage {
                inner: preprocessed,
                is_spam,
            },
        },
    );

    Ok(iter)
}

/// AI guard logs are provided in a certain format that we a thriving to parse.
fn filter_ai_guard_logs<I>(
    iter: I,
    chat_id: i64,
) -> impl Iterator<Item = Result<Option<(u64, tg_antispam::PreprocessedMessage)>, snafu::Whatever>>
where
    I: IntoIterator<Item = TgMessage>,
{
    iter.into_iter().skip(3).map(move |tg_message| {
        let message_id = tg_message.id;
        let from = tg_message
            .from
            .with_whatever_context(|| format!("Missing \"from\", #{message_id}"))?;
        let from_id = tg_message
            .from_id
            .with_whatever_context(|| format!("Missing from_id, #{message_id}"))?;

        let mut entities = tg_message.text_entities.into_iter();
        let channel_source = entities
            .next()
            .with_whatever_context(|| format!("Missing source channel, #{message_id}"))?;
        if !matches!(channel_source.type_, TgEntityType::Mention) {
            return Ok(None);
        }
        let combined_text = entities
            .next()
            .with_whatever_context(|| format!("Missing the text field, #{message_id}"))?;
        snafu::ensure_whatever!(
            matches!(combined_text.type_, TgEntityType::Plain),
            "Unexpected text field type message #{message_id}: {:?}",
            combined_text.type_
        );

        let text = std::iter::once(combined_text.text)
            .chain(entities.map(|entity| entity.text))
            .collect::<String>();

        let text = parse_ai_guard(&text)
            .with_whatever_context(|_| format!("Can't parse message #{message_id}"))?;
        Ok(Some((
            message_id as u64,
            tg_antispam::PreprocessedMessage {
                text,
                from,
                from_id,
                chat_id,
                time: tg_message.date,
                forwarded_from: tg_message.forwarded_from,
            },
        )))
    })
}

fn parse_ai_guard(text: &str) -> Result<String, snafu::Whatever> {
    let mut parts = text.split_inclusive('\n');
    let _chat_name = parts.next().whatever_context("Chat name missing")?;
    let _user_name = parts.next().whatever_context("User name missing")?;

    let old_format = [
        "Шанс, что это спам",
        "Уже видел... точно спам",
        "Помечено как спам по результатам",
        "Ой ой, Вы очень плохой",
        "Помечено как спам модератором",
        "Already seen... definitely spam",
        "Chance that this is spam",
        "Probabilidad de que esto sea spam",
    ];
    let new_format = ["Наспамил", "Spammed", "Ha enviado spam"];

    let spam_field = parts
        .next()
        .whatever_context("Missing spam description field")?;

    if old_format.iter().any(|start| spam_field.starts_with(start)) {
        // This is the "old" format. The rest is the spam message!
        Ok(parts.join("\n"))
    } else if new_format.iter().any(|start| spam_field.starts_with(start)) {
        // This is the "new" format.
        let first_text = parts.next().whatever_context("Missing the text field")?;
        let first_text = first_text
            .strip_prefix("Текст:")
            .or_else(|| first_text.strip_prefix("Text:"))
            .or_else(|| first_text.strip_prefix("Texto:"))
            .with_whatever_context(|| format!("Missing the prefix in {first_text:?}"))?;
        Ok(std::iter::once(first_text)
            .chain(parts.with_position().filter_map(|(pos, text)| {
                match pos {
                    itertools::Position::First | itertools::Position::Middle => Some(text),
                    itertools::Position::Last | itertools::Position::Only => {
                        // Skip the last text.
                        assert!(
                            text.starts_with("Шанс, что это спам")
                                || text.starts_with("Уже видел")
                                || text.starts_with("Помечено как спам")
                                || text.starts_with("Ой ой, Вы очень плохой человек")
                                || text.starts_with("Marked as spam ")
                                || text.starts_with("Already seen... definitely spam")
                                || text.starts_with("Chance that this is spam")
                                || text.starts_with("Probabilidad de que esto"),
                            "Unexpected ending: {text:?}",
                        );
                        None
                    }
                }
            }))
            .join("\n"))
    } else {
        snafu::whatever!("Unknown format. Text: {text}");
    }
}
