//! BERT model implementation.
//!
//! This file is copied from the `rust-bert` crate.

use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashMap,
};

use crate::{
    activations::Activation,
    attention::{BertAttention, BertIntermediate},
    dropout::Dropout,
    Freeze,
};

use serde::{Deserialize, Serialize};
use tch::{
    self,
    nn::{self, embedding, EmbeddingConfig},
    Device, Kind, Tensor,
};

#[derive(thiserror::Error, Debug)]
pub enum RustBertError {
    #[error("Value error: {0}")]
    ValueError(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # BERT model configuration
/// Defines the BERT model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct BertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
}

/// # BertEmbedding trait (for use in BertModel or RoBERTaModel)
/// Defines an interface for the embedding layers in BERT-based models
pub trait BertEmbedding {
    fn new<'p, P>(p: P, config: &BertConfig) -> Self
    where
        P: Borrow<nn::Path<'p>>;

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, RustBertError>;
}

#[derive(Debug)]
/// # BertEmbeddings implementation for BERT model
/// Implementation of the `BertEmbedding` trait for BERT models
pub struct BertEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl Freeze for BertEmbeddings {
    fn set_freeze(&self, freeze: bool) {
        self.word_embeddings.set_freeze(freeze);
        self.position_embeddings.set_freeze(freeze);
        self.token_type_embeddings.set_freeze(freeze);
        self.layer_norm.set_freeze(freeze);
    }
}

impl BertEmbedding for BertEmbeddings {
    /// Build a new `BertEmbeddings`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BertEmbeddings model
    /// * `config` - `BertConfig` object defining the model architecture and vocab/hidden size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertEmbedding, BertEmbeddings};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert_embeddings = BertEmbeddings::new(&p.root() / "bert_embeddings", &config);
    /// ```
    fn new<'p, P>(p: P, config: &BertConfig) -> BertEmbeddings
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embedding_config = EmbeddingConfig {
            padding_idx: 0,
            ..Default::default()
        };

        let word_embeddings: nn::Embedding = embedding(
            p / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            embedding_config,
        );

        let position_embeddings: nn::Embedding = embedding(
            p / "position_embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default(),
        );

        let token_type_embeddings: nn::Embedding = embedding(
            p / "token_type_embeddings",
            config.type_vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-12,
            ..Default::default()
        };
        let layer_norm: nn::LayerNorm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);
        let dropout: Dropout = Dropout::new(config.hidden_dropout_prob);
        BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    /// Forward pass through the embedding layer
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see *input_embeds*)
    /// * `token_type_ids` -Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see *input_ids*)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `embedded_output` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertEmbeddings, BertEmbedding};
    /// # use tch::{nn, Device, Tensor, no_grad};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # use tch::kind::Kind::Int64;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let vocab_path = Path::new("path/to/vocab.txt");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_embeddings = BertEmbeddings::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let embedded_output = no_grad(|| {
    ///     bert_embeddings
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, RustBertError> {
        let (calc_input_embeddings, input_shape, _) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.word_embeddings)?;

        let input_embeddings =
            input_embeds.unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap());
        let seq_length = input_embeddings.size()[1];

        let calc_position_ids = if position_ids.is_none() {
            Some(
                Tensor::arange(seq_length, (Kind::Int64, input_embeddings.device()))
                    .unsqueeze(0)
                    .expand(&input_shape, true),
            )
        } else {
            None
        };

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(
                &input_shape,
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| calc_token_type_ids.as_ref().unwrap());

        let position_embeddings = position_ids.apply(&self.position_embeddings);
        let token_type_embeddings = token_type_ids.apply(&self.token_type_embeddings);

        let input_embeddings: Tensor =
            input_embeddings + position_embeddings + token_type_embeddings;
        Ok(input_embeddings
            .apply(&self.layer_norm)
            .apply_t(&self.dropout, train))
    }
}

fn process_ids_embeddings_pair(
    input_ids: Option<&Tensor>,
    input_embeddings: Option<&Tensor>,
    embeddings_matrix: &tch::nn::Embedding,
) -> Result<(Option<Tensor>, Vec<i64>, Device), RustBertError> {
    Ok(match (input_ids, input_embeddings) {
        (Some(_), Some(_)) => {
            return Err(RustBertError::ValueError(
                "Only one of input ids or input embeddings may be set".into(),
            ));
        }
        (Some(input_value), None) => (
            Some(input_value.apply(embeddings_matrix)),
            input_value.size(),
            input_value.device(),
        ),
        (None, Some(embeds)) => {
            let size = vec![embeds.size()[0], embeds.size()[1]];
            (None, size, embeds.device())
        }
        (None, None) => {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        }
    })
}

fn get_shape_and_device_from_ids_embeddings_pair(
    input_ids: Option<&Tensor>,
    input_embeddings: Option<&Tensor>,
) -> Result<(Vec<i64>, Device), RustBertError> {
    Ok(match (input_ids, input_embeddings) {
        (Some(_), Some(_)) => {
            return Err(RustBertError::ValueError(
                "Only one of input ids or input embeddings may be set".into(),
            ));
        }
        (Some(input_value), None) => (input_value.size(), input_value.device()),
        (None, Some(embeds)) => {
            let size = vec![embeds.size()[0], embeds.size()[1]];
            (size, embeds.device())
        }
        (None, None) => {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        }
    })
}

/// # BERT Base model
/// Base architecture for BERT models. Task-specific models will be built from this common base model
/// It is made of the following blocks:
/// - `embeddings`: `token`, `position` and `segment_id` embeddings
/// - `encoder`: Encoder (transformer) made of a vector of layers. Each layer is made of a self-attention layer, an intermediate (linear) and output (linear + layer norm) layers
/// - `pooler`: linear layer applied to the first element of the sequence (*MASK* token)
/// - `is_decoder`: Flag indicating if the model is used as a decoder. If set to true, a causal mask will be applied to hide future positions that should not be attended to.
pub struct BertModel<T: BertEmbedding = BertEmbeddings> {
    embeddings: T,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
    is_decoder: bool,
}

impl<T: BertEmbedding + Freeze> Freeze for BertModel<T> {
    fn set_freeze(&self, freeze: bool) {
        self.embeddings.set_freeze(freeze);
        self.encoder.set_freeze(freeze);
        self.pooler.set_freeze(freeze);
    }
}

/// Defines the implementation of the BertModel. The BERT model shares many similarities with RoBERTa, main difference being the embeddings.
/// Therefore the forward pass of the model is shared and the type of embedding used is abstracted away. This allows to create
/// `BertModel<RobertaEmbeddings>` or `BertModel<BertEmbeddings>` for each model type.
impl<T: BertEmbedding> BertModel<T> {
    /// Build a new `BertModel`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture and decoder status
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertEmbeddings, BertModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert: BertModel<BertEmbeddings> = BertModel::new(&p.root() / "bert", &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertModel<T>
    where
        P: Borrow<nn::Path<'p>>,
    {
        Self::new_with_optional_pooler(p, config, true)
    }

    /// Returns the BERT encoder layers.
    pub fn encoder_layers(
        &self,
    ) -> impl DoubleEndedIterator<Item = &'_ BertLayer> + ExactSizeIterator {
        self.encoder.layers.iter()
    }

    /// Build a new `BertModel` with an optional Pooling layer
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture and decoder status
    /// * `add_pooling_layer` - Enable/Disable an optional pooling layer at the end of the model
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertEmbeddings, BertModel};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let bert: BertModel<BertEmbeddings> =
    ///     BertModel::new_with_optional_pooler(&p.root() / "bert", &config, false);
    /// ```
    pub fn new_with_optional_pooler<'p, P>(
        p: P,
        config: &BertConfig,
        add_pooling_layer: bool,
    ) -> BertModel<T>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let is_decoder = config.is_decoder.unwrap_or(false);
        let embeddings = { T::new(p / "embeddings", config) };
        let encoder = BertEncoder::new(p / "encoder", config);

        let pooler = {
            if add_pooling_layer {
                Some(BertPooler::new(p / "pooler", config))
            } else {
                None
            }
        };

        BertModel {
            embeddings,
            encoder,
            pooler,
            is_decoder,
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `pooled_output` - `Tensor` of shape (*batch size*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertModel, BertConfig, BertEmbeddings};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model: BertModel<BertEmbeddings> = BertModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&mask),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             None,
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<BertModelOutput, RustBertError> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

        let calc_mask = Tensor::ones(&input_shape, (Kind::Int8, device));
        let mask = mask.unwrap_or(&calc_mask);

        let extended_attention_mask = match mask.dim() {
            3 => mask.unsqueeze(1),
            2 => {
                if self.is_decoder {
                    let seq_ids = Tensor::arange(input_shape[1], (Kind::Int8, device));
                    let causal_mask = seq_ids.unsqueeze(0).unsqueeze(0).repeat([
                        input_shape[0],
                        input_shape[1],
                        1,
                    ]);
                    let causal_mask = causal_mask.le_tensor(&seq_ids.unsqueeze(0).unsqueeze(-1));
                    causal_mask * mask.unsqueeze(1).unsqueeze(1)
                } else {
                    mask.unsqueeze(1).unsqueeze(1)
                }
            }
            _ => {
                return Err(RustBertError::ValueError(
                    "Invalid attention mask dimension, must be 2 or 3".into(),
                ));
            }
        };

        let embedding_output = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            input_embeds,
            train,
        )?;

        let extended_attention_mask: Tensor = ((extended_attention_mask
            .ones_like()
            .bitwise_xor_tensor(&extended_attention_mask))
            * -10000.0)
            .to_kind(embedding_output.kind());

        let encoder_extended_attention_mask: Option<Tensor> =
            if self.is_decoder & encoder_hidden_states.is_some() {
                let encoder_hidden_states = encoder_hidden_states.as_ref().unwrap();
                let encoder_hidden_states_shape = encoder_hidden_states.size();
                let encoder_mask = match encoder_mask {
                    Some(value) => value.copy(),
                    None => Tensor::ones(
                        [
                            encoder_hidden_states_shape[0],
                            encoder_hidden_states_shape[1],
                        ],
                        (Kind::Int8, device),
                    ),
                };
                match encoder_mask.dim() {
                    2 => Some(encoder_mask.unsqueeze(1).unsqueeze(1)),
                    3 => Some(encoder_mask.unsqueeze(1)),
                    _ => {
                        return Err(RustBertError::ValueError(
                            "Invalid attention mask dimension, must be 2 or 3".into(),
                        ));
                    }
                }
            } else {
                None
            };

        let encoder_output = self.encoder.forward_t(
            &embedding_output,
            Some(&extended_attention_mask),
            encoder_hidden_states,
            encoder_extended_attention_mask.as_ref(),
            train,
        );

        let pooled_output = self
            .pooler
            .as_ref()
            .map(|pooler| pooler.forward(&encoder_output.hidden_state));

        Ok(BertModelOutput {
            hidden_state: encoder_output.hidden_state,
            pooled_output,
            all_hidden_states: encoder_output.all_hidden_states,
            all_attentions: encoder_output.all_attentions,
        })
    }
}

/// Container for the BERT model output.
pub struct BertModelOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Pooled output (hidden state for the first token)
    pub pooled_output: Option<Tensor>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

/// # BERT Encoder
/// Encoder used in BERT models.
/// It is made of a Vector of `BertLayer` through which hidden states will be passed. The encoder can also be
/// used as a decoder (with cross-attention) if `encoder_hidden_states` are provided.
pub struct BertEncoder {
    output_attentions: bool,
    output_hidden_states: bool,
    layers: Vec<BertLayer>,
}

impl Freeze for BertEncoder {
    fn set_freeze(&self, freeze: bool) {
        self.layers
            .iter()
            .for_each(|layer| layer.set_freeze(freeze))
    }
}

impl BertEncoder {
    /// Build a new `BertEncoder`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertEncoder};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let encoder: BertEncoder = BertEncoder::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertEncoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "layer";
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);

        let mut layers: Vec<BertLayer> = vec![];
        for layer_index in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(&p / layer_index, config));
        }

        BertEncoder {
            output_attentions,
            output_hidden_states,
            layers,
        }
    }

    /// Forward pass through the encoder
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertEncoderOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `all_attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertEncoder};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// let encoder: BertEncoder = BertEncoder::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, hidden_size) = (64, 128, 512);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, hidden_size],
    ///     (Kind::Float, device),
    /// );
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int8, device));
    ///
    /// let encoder_output =
    ///     no_grad(|| encoder.forward_t(&input_tensor, Some(&mask), None, None, false));
    /// ```
    pub fn forward_t(
        &self,
        input: &Tensor,
        mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> BertEncoderOutput {
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        let mut hidden_state = None::<Tensor>;
        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let layer_output = if let Some(hidden_state) = &hidden_state {
                layer.forward_t(
                    hidden_state,
                    mask,
                    encoder_hidden_states,
                    encoder_mask,
                    train,
                )
            } else {
                layer.forward_t(input, mask, encoder_hidden_states, encoder_mask, train)
            };

            hidden_state = Some(layer_output.hidden_state);
            attention_weights = layer_output.attention_weights;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut attention_weights.unwrap()));
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(hidden_state.as_ref().unwrap().copy());
            };
        }

        BertEncoderOutput {
            hidden_state: hidden_state.unwrap(),
            all_hidden_states,
            all_attentions,
        }
    }
}

/// # BERT Layer
/// Layer used in BERT encoders.
/// It is made of the following blocks:
/// - `attention`: self-attention `BertAttention` layer
/// - `cross_attention`: (optional) cross-attention `BertAttention` layer (if the model is used as a decoder)
/// - `is_decoder`: flag indicating if the model is used as a decoder
/// - `intermediate`: `BertIntermediate` intermediate layer
/// - `output`: `BertOutput` output layer
pub struct BertLayer {
    attention: BertAttention,
    is_decoder: bool,
    cross_attention: Option<BertAttention>,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl Freeze for BertLayer {
    fn set_freeze(&self, freeze: bool) {
        self.attention.set_freeze(freeze);
        self.cross_attention.set_freeze(freeze);
        self.intermediate.set_freeze(freeze);
        self.output.set_freeze(freeze);
    }
}

impl BertLayer {
    /// Build a new `BertLayer`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertLayer};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let layer: BertLayer = BertLayer::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let attention = BertAttention::new(p / "attention", config);
        let (is_decoder, cross_attention) = match config.is_decoder {
            Some(value) => {
                if value {
                    (
                        value,
                        Some(BertAttention::new(p / "cross_attention", config)),
                    )
                } else {
                    (value, None)
                }
            }
            None => (false, None),
        };

        let intermediate = BertIntermediate::new(p / "intermediate", config);
        let output = BertOutput::new(p / "output", config);

        BertLayer {
            attention,
            is_decoder,
            cross_attention,
            intermediate,
            output,
        }
    }

    /// Forward pass through the layer
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    /// * `mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `encoder_hidden_states` - Optional encoder hidden state of shape (*batch size*, *encoder_sequence_length*, *hidden_size*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used in the cross-attention layer as keys and values (query from the decoder).
    /// * `encoder_mask` - Optional encoder attention mask of shape (*batch size*, *encoder_sequence_length*). If the model is defined as a decoder and the `encoder_hidden_states` is not None, used to mask encoder values. Positions with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `BertLayerOutput` containing:
    ///   - `hidden_state` - `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `attention_scores` - `Option<Tensor>` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `cross_attention_scores` - `Option<Tensor>` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertLayer};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// let layer: BertLayer = BertLayer::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, hidden_size) = (64, 128, 512);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, hidden_size],
    ///     (Kind::Float, device),
    /// );
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    ///
    /// let layer_output = no_grad(|| layer.forward_t(&input_tensor, Some(&mask), None, None, false));
    /// ```
    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> BertLayerOutput {
        let (attention_output, attention_weights) =
            self.attention
                .forward_t(hidden_states, mask, None, None, train);

        let (attention_output, attention_scores, cross_attention_scores) =
            if self.is_decoder & encoder_hidden_states.is_some() {
                let (attention_output, cross_attention_weights) =
                    self.cross_attention.as_ref().unwrap().forward_t(
                        &attention_output,
                        mask,
                        encoder_hidden_states,
                        encoder_mask,
                        train,
                    );
                (attention_output, attention_weights, cross_attention_weights)
            } else {
                (attention_output, attention_weights, None)
            };

        let output = self.intermediate.forward(&attention_output);
        let output = self.output.forward_t(&output, &attention_output, train);

        BertLayerOutput {
            hidden_state: output,
            attention_weights: attention_scores,
            cross_attention_weights: cross_attention_scores,
        }
    }
}

/// # BERT Pooler
/// Pooler used in BERT models.
/// It is made of a fully connected layer which is applied to the first sequence element.
pub struct BertPooler {
    lin: nn::Linear,
}

impl Freeze for BertPooler {
    fn set_freeze(&self, freeze: bool) {
        self.lin.set_freeze(freeze)
    }
}

impl BertPooler {
    /// Build a new `BertPooler`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BERT model
    /// * `config` - `BertConfig` object defining the model architecture
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::bert::{BertConfig, BertPooler};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use tch::{nn, Device};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BertConfig::from_file(config_path);
    /// let pooler: BertPooler = BertPooler::new(&p.root(), &config);
    /// ```
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertPooler
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let lin = nn::linear(
            p / "dense",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        BertPooler { lin }
    }

    /// Forward pass through the pooler
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - input tensor of shape (*batch size*, *sequence_length*, *hidden_size*).
    ///
    /// # Returns
    ///
    /// * `Tensor` of shape (*batch size*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertConfig, BertPooler};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// let pooler: BertPooler = BertPooler::new(&vs.root(), &config);
    /// let (batch_size, sequence_length, hidden_size) = (64, 128, 512);
    /// let input_tensor = Tensor::rand(
    ///     &[batch_size, sequence_length, hidden_size],
    ///     (Kind::Float, device),
    /// );
    ///
    /// let pooler_output = no_grad(|| pooler.forward(&input_tensor));
    /// ```
    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        hidden_states.select(1, 0).apply(&self.lin).tanh()
    }
}

/// Container for the BERT layer output.
pub struct BertLayerOutput {
    /// Hidden states
    pub hidden_state: Tensor,
    /// Self attention scores
    pub attention_weights: Option<Tensor>,
    /// Cross attention scores
    pub cross_attention_weights: Option<Tensor>,
}

/// Container for the BERT encoder output.
pub struct BertEncoderOutput {
    /// Last hidden states from the model
    pub hidden_state: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

pub struct BertOutput {
    lin: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl Freeze for BertOutput {
    fn set_freeze(&self, freeze: bool) {
        self.lin.set_freeze(freeze);
        self.layer_norm.set_freeze(freeze);
    }
}

impl BertOutput {
    pub fn new<'p, P>(p: P, config: &BertConfig) -> BertOutput
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let lin = nn::linear(
            p / "dense",
            config.intermediate_size,
            config.hidden_size,
            Default::default(),
        );
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-12,
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        BertOutput {
            lin,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let hidden_states: Tensor =
            input_tensor + hidden_states.apply(&self.lin).apply_t(&self.dropout, train);
        hidden_states.apply(&self.layer_norm)
    }
}
