use serde::Deserialize;
use tg_antispam::PreprocessedMessage;

/// Telegram export data.
#[derive(Debug, Deserialize)]
pub struct TelegramExportData {
    /// Chat ID.
    pub id: i64,
    /// Messages in the chat.
    pub messages: Vec<TgMessage>,
}

/// A single telegram messages.
#[derive(Debug, Deserialize, Clone)]
pub struct TgMessage {
    pub id: i64,
    #[serde(rename = "date_unixtime", deserialize_with = "parse_date_unixtime")]
    pub date: time::OffsetDateTime,
    pub text_entities: Vec<TelegramTextEntity>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub forwarded_from: Option<Option<String>>,
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub from_id: Option<String>,
}

/// Any value that is present is considered Some value, including null.
fn deserialize_some<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Deserialize::deserialize(deserializer).map(Some)
}

/// Parses a stringified unix date time.
fn parse_date_unixtime<'de, D>(deserializer: D) -> Result<time::OffsetDateTime, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Either<'a> {
        A(&'a str),
        B(String),
    }
    let either = Either::deserialize(deserializer)?;
    let s = match &either {
        Either::A(s) => s,
        Either::B(s) => s.as_str(),
    };
    let unix_date_time: i64 = s.parse().map_err(<D::Error as serde::de::Error>::custom)?;
    time::OffsetDateTime::from_unix_timestamp(unix_date_time)
        .map_err(<D::Error as serde::de::Error>::custom)
}

/// A telegram text entity.
#[derive(Debug, Deserialize, Clone)]
pub struct TelegramTextEntity {
    #[serde(rename = "type")]
    pub type_: TgEntityType,
    pub text: String,
    #[serde(default)]
    pub href: Option<String>,
}

/// Telegram entity type.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TgEntityType {
    BankCard,
    Blockquote,
    Bold,
    BotCommand,
    Cashtag,
    Code,
    CustomEmoji,
    Email,
    Hashtag,
    Italic,
    Link,
    Mention,
    MentionName,
    Phone,
    Plain,
    Pre,
    Spoiler,
    Strikethrough,
    TextLink,
    Underline,
    #[serde(other)]
    Other,
}

/// Preprocesses the incoming message.
pub fn preprocess_message(raw: TgMessage, chat_id: i64) -> Option<PreprocessedMessage> {
    let from = raw.from?;
    let from_id = raw.from_id?;
    let mut whole_text = String::new();
    for TelegramTextEntity { type_, text, href } in raw.text_entities {
        if type_ == TgEntityType::Link {
            let filtered = filter_url(&text);
            whole_text.push_str(&filtered);
        } else if type_ == TgEntityType::TextLink {
            let url = filter_url(href.as_deref().unwrap_or_default());
            use std::fmt::Write as _;
            write!(whole_text, "{text} ({url})").expect("Writing to a String never fails");
        } else if text.contains('\n') {
            let text = text.replace('\n', ". ");
            whole_text.push_str(&text);
        } else {
            whole_text.push_str(&text);
        }
    }
    Some(PreprocessedMessage {
        text: whole_text,
        from,
        from_id,
        time: raw.date,
        chat_id,
        forwarded_from: raw.forwarded_from,
    })
}

/// Filters out URL by removing colons, dots and stuff.
pub fn filter_url(input: &str) -> String {
    let decoded_result = urlencoding::decode(input);
    let input = decoded_result.as_deref().unwrap_or(input);
    input.replace([':', '?', '.', '#', '&', '%', '='], " ")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_url_strip() {
        assert_eq!(
            filter_url("https://youtube.com/aaa"),
            "https //youtube com/aaa"
        );
        assert_eq!(filter_url("tg://join?invite=abcd"), "tg //join invite abcd");
        assert_eq!(filter_url("https://vk.com"), "https //vk com");
        assert_eq!(filter_url("https://t.me"), "https //t me");
        assert_eq!(
            filter_url("https://t.me/some_name/id"),
            "https //t me/some_name/id"
        );
        assert_eq!(
            filter_url("https://t.me/some_name/"),
            "https //t me/some_name/"
        );
        assert_eq!(
            filter_url("https://t.me/some_name"),
            "https //t me/some_name"
        );
        assert_eq!(filter_url("https://t.me/"), "https //t me/");
    }
}
