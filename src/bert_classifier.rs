//! A BERT-like encoder.

use snafu::{ResultExt, Snafu};
use tch::{nn::Module, IndexOp, Tensor};

use crate::bert::BertConfig;

type BertModel = crate::bert::BertModel<crate::bert::BertEmbeddings>;

/// A sentence classifier.
pub struct BertClassifier {
    bert: BertModel,
    classification_layer: tch::nn::Linear,
    device: tch::Device,
    pad_token: i64,
}

/// Feed forward error.
#[derive(Debug, Snafu)]
pub enum ForwardError {
    /// Error on the BERT layers.
    #[snafu(display("Error on the BERT layers"))]
    BertForward {
        /// Source error.
        source: crate::bert::RustBertError,
    },
}

/// Output of the sentence classifier.
pub struct ClassifierOutput {
    /// The probability that a sentence is a spam.
    pub unnormalized_probabilities: Tensor,
}

impl BertClassifier {
    /// Initializes a [`BertClassifier`] model for the inference purposes.
    pub fn inference(
        bert_config: &BertConfig,
        var_store: &tch::nn::Path<'_>,
        pad_token: i64,
    ) -> Self {
        let bert_model =
            crate::bert::BertModel::<crate::bert::BertEmbeddings>::new_with_optional_pooler(
                var_store / "bert",
                bert_config,
                false,
            );
        Self::new(bert_model, bert_config.hidden_size, var_store, pad_token)
    }

    /// Initializes the named sentence classifier with a possibly pre-trained
    /// BERT model.
    pub fn new(
        bert: BertModel,
        bert_hidden_size: i64,
        var_store: &tch::nn::Path<'_>,
        pad_token: i64,
    ) -> Self {
        let device = var_store.device();
        let layer1 = tch::nn::linear(
            var_store / "classification" / "classifier1",
            bert_hidden_size,
            1,
            tch::nn::LinearConfig::default(),
        );
        Self {
            bert,
            classification_layer: layer1,
            device,
            pad_token,
        }
    }

    /// Returns the reference to the underlying BERT model.
    pub fn bert(&self) -> &BertModel {
        &self.bert
    }

    /// Predicts if the input sentence in a spam.
    pub fn predict<I, Tokens>(&self, input: I) -> Result<Tensor, ForwardError>
    where
        I: IntoIterator<Item = Tokens>,
        Tokens: AsRef<[i64]>,
    {
        let ClassifierOutput {
            unnormalized_probabilities,
        } = self.forward(input, false)?;
        let probabilities = unnormalized_probabilities.sigmoid();
        Ok(probabilities)
    }

    /// Runs a forward pass on the model for the given tokenized sentences.
    ///
    /// For each input word the model outputs a vector of unnormalized weights
    /// which should be softmaxed to obtain probabilities of a word to belong to
    /// a certain class.
    pub fn forward<I, Tokens>(
        &self,
        input: I,
        train: bool,
    ) -> Result<ClassifierOutput, ForwardError>
    where
        I: IntoIterator<Item = Tokens>,
        Tokens: AsRef<[i64]>,
    {
        let mut batch_tokens = vec![];
        for tokens in input {
            let tokens = Tensor::from_slice(tokens.as_ref());
            batch_tokens.push(tokens);
        }
        let tokens =
            Tensor::pad_sequence(&batch_tokens, true, self.pad_token as f64).to(self.device);
        let mask = tokens.not_equal(self.pad_token);
        debug_assert_eq!(mask.device(), self.device);

        let bert_embeddings = self
            .bert
            .forward_t(
                Some(&tokens),
                Some(&mask),
                None,
                None,
                None,
                None,
                None,
                train,
            )
            .context(BertForwardSnafu)?
            .hidden_state;
        // Take embeddings of the first (CLS) token.
        let sentence_embeddings = bert_embeddings.i((.., 0, ..));

        let output_tensor = self.classification_layer.forward(&sentence_embeddings);
        Ok(ClassifierOutput {
            unnormalized_probabilities: output_tensor,
        })
    }

    /// Feed forwards the input and calculates the loss using a cross entropy.
    pub fn train<'a, I>(
        &self,
        input: I,
        pos_weight: Option<&Tensor>,
    ) -> Result<Tensor, ForwardError>
    where
        I: IntoIterator<Item = (&'a [i64], bool)>,
    {
        let (forward_input, tags_per_sentence): (Vec<_>, Vec<_>) = input
            .into_iter()
            .map(|(tokens, is_spam)| (tokens, f32::from(is_spam)))
            .unzip();

        let tags_per_sentence = Tensor::from_slice(&tags_per_sentence).to(self.device);

        let output = self.forward(forward_input, true)?;
        let loss = self.loss(&output, &tags_per_sentence, pos_weight);
        Ok(loss)
    }

    /// Computes the loss between the model's output and expected tags (as a
    /// number of a class) of each word that has been fed to the model.
    fn loss(
        &self,
        model_output: &ClassifierOutput,
        expected_tags: &Tensor,
        pos_weight: Option<&Tensor>,
    ) -> Tensor {
        let output = model_output.unnormalized_probabilities.squeeze_dim(1);
        output.binary_cross_entropy_with_logits(
            expected_tags,
            None,
            pos_weight,
            tch::Reduction::Mean,
        )
    }
}
