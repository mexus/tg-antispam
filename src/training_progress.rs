//! Training and evaluation progress information.

use camino::Utf8Path;
use hdrhistogram::Histogram;
use owo_colors::{AnsiColors, OwoColorize};
use snafu::{ResultExt, Snafu};
use tch::Tensor;

/// Threshold histograms.
#[derive(Debug, Clone)]
pub struct ThresholdInfo {
    /// What "spam" messages are labeled as.
    pub spam: Histogram<u32>,
    /// What "ham" messages are labeled as.
    pub not_spam: Histogram<u32>,
}

impl Default for ThresholdInfo {
    fn default() -> Self {
        let hist = Histogram::<u32>::new_with_max(100, 2).expect("Must be OK");
        Self {
            spam: hist.clone(),
            not_spam: hist,
        }
    }
}

impl ThresholdInfo {
    /// Records spam prediction.
    pub fn record_prediction(&mut self, is_spam: bool, prediction: f32) {
        assert!(
            (0f32..=1f32).contains(&prediction),
            "prediction out of range: {prediction}"
        );

        let percent = (prediction * 100.).round() as u64;
        if is_spam {
            self.spam.record(percent).expect("Must be fine");
        } else {
            self.not_spam.record(percent).expect("Must be fine");
        }
    }
}

impl std::fmt::Display for ThresholdInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let thresholds = [
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
        ];

        let (precisions, recalls): (Vec<_>, Vec<_>) =
            self.precision_and_recall_list(thresholds).unzip();

        // Header.
        writeln!(f, "Precision and recall with respect to a threshold")?;
        const FIRST_WIDTH: usize = 9; // First cell width
        const WIDTH: usize = 7; // cell width

        write!(f, "|{: >FIRST_WIDTH$}|", "")?;
        thresholds.into_iter().try_for_each(|threshold| {
            let threshold = threshold as f32 / 100.;
            write!(f, "{: ^WIDTH$}|", format!("t={threshold:.2}"))
        })?;
        writeln!(f)?;

        // Delimiter.
        write!(f, "|{:->FIRST_WIDTH$}|", "")?;
        thresholds
            .into_iter()
            .try_for_each(|_threshold| write!(f, "{:-^WIDTH$}|", ""))?;
        writeln!(f)?;

        // Recall.
        write!(f, "|{:^FIRST_WIDTH$}|", "Precision")?;
        recalls
            .into_iter()
            .try_for_each(|recall| write!(f, "{:^WIDTH$}|", format!("{:.2}%", recall * 100.)))?;
        writeln!(f)?;

        // Precision.
        write!(f, "|{:^FIRST_WIDTH$}|", "Recall")?;
        precisions.into_iter().try_for_each(|precision| {
            write!(f, "{:^WIDTH$}|", format!("{:.2}%", precision * 100.))
        })?;
        writeln!(f)?;

        // F1.
        write!(f, "|{:^FIRST_WIDTH$}|", "F1")?;
        self.f_beta_list(thresholds, 1.0)
            .try_for_each(|f1| write!(f, "{:^WIDTH$}|", format!("{:.2}%", f1 * 100.)))?;
        writeln!(f)?;

        // F 0.5
        write!(f, "|{:^FIRST_WIDTH$}|", "F0.5")?;
        self.f_beta_list(thresholds, 0.5)
            .try_for_each(|f1| write!(f, "{:^WIDTH$}|", format!("{:.2}%", f1 * 100.)))?;
        writeln!(f)?;

        // F 2
        write!(f, "|{:^FIRST_WIDTH$}|", "F2")?;
        self.f_beta_list(thresholds, 2.)
            .try_for_each(|f1| write!(f, "{:^WIDTH$}|", format!("{:.2}%", f1 * 100.)))?;
        writeln!(f)?;

        Ok(())
    }
}

impl ThresholdInfo {
    /// Finds the best threshold that gives the best F-β score.
    pub fn best(&self, beta: f64) -> (u64, f64) {
        let range = 0..=99;
        range
            .clone()
            .zip(self.f_beta_list(range, beta))
            .max_by(|(_, x), (_, y)| x.total_cmp(y))
            .expect("Not empty")
    }

    /// Returns precision and recall at the given threshold.
    pub fn precision_and_recall(&self, threshold: f64) -> (f64, f64) {
        let threshold = (threshold * 100.).round() as u64;

        // The amount of spam samples.
        let real_spam_count = self.spam.len();
        // The amount of non-spam samples.
        let real_ham_count = self.not_spam.len();

        // The probability that a message is tagged as a spam while begin a
        // spam. This is "true positives".
        let spam_success = 1. - self.spam.quantile_below(threshold);
        let true_positives = (real_spam_count as f64 * spam_success).round() as u64;

        // Samples that are spam, but incorrectly tagged as ham.
        let spam_failure = self.spam.quantile_below(threshold);
        let false_negatives = (real_spam_count as f64 * spam_failure).round() as u64;

        // The probability that a message is tagged as a ham while begin a
        // ham. This is "true negatives".
        let ham_success = self.not_spam.quantile_below(threshold);
        let _true_negatives = (real_ham_count as f64 * ham_success).round() as u64;

        // Samples that are ham, but incorrectly tagged as spam.
        let ham_failure = 1. - self.not_spam.quantile_below(threshold);
        let false_positives = (real_ham_count as f64 * ham_failure).round() as u64;

        let all_retrieved_instances = true_positives + false_positives;
        let relevant_retrieved_instances = true_positives;
        let all_relevant_instances = false_negatives + true_positives;

        let precision = relevant_retrieved_instances as f64 / all_retrieved_instances as f64;
        let recall = relevant_retrieved_instances as f64 / all_relevant_instances as f64;

        (precision, recall)
    }

    /// Returns the Fβ score.
    ///
    /// β is a positive real factor. Recall is considered β times as important
    /// as precision.
    ///
    /// Examples:
    /// * β = 2 weighs recall twice as important as precision;
    /// * β = 1 weights recall and precision equally;
    /// * β = 0.5 weights precision twice as important as recall.
    pub fn fn_score(&self, threshold: f64, beta: f64) -> f64 {
        let (precision, recall) = self.precision_and_recall(threshold);
        let beta_squared = beta * beta;
        (1. + beta_squared) / (beta_squared * precision + recall)
    }

    fn f_beta_list<'a, I>(&'a self, thresholds: I, beta: f64) -> impl Iterator<Item = f64> + 'a
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: 'a,
    {
        let beta_squared = beta * beta;
        self.precision_and_recall_list(thresholds)
            .map(move |(precision, recall)| {
                (1. + beta_squared) * precision * recall / (beta_squared * precision + recall)
            })
    }

    /// Computes precision and recall for each provided threshold.
    fn precision_and_recall_list<'a, I>(
        &'a self,
        thresholds: I,
    ) -> impl Iterator<Item = (f64, f64)> + 'a
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: 'a,
    {
        thresholds
            .into_iter()
            .map(|threshold| self.precision_and_recall(threshold as f64 / 100.))
    }
}

/// Training and evaluation progress information accumulator and printer.
pub struct EpochReportRecorder {
    /// Loss to amount of training samples.
    loss: Vec<(usize, f32)>,
    /// How learning rate changes within the epoch.
    learning_rate: Vec<(usize, f64)>,
    /// The amount of training samples (un-batched).
    samples_count: usize,
    /// Epoch number.
    epoch: usize,
    /// The amount of processed data samples so far.
    samples_processed: usize,
}

impl EpochReportRecorder {
    /// Initializes an empty progress information.
    pub fn new(epoch: usize, samples_count: usize) -> Self {
        Self {
            samples_count,
            epoch,
            loss: vec![],
            learning_rate: vec![],
            samples_processed: 0,
        }
    }

    /// Records a feed forward loss on the given amount of samples.
    pub fn record(&mut self, samples: usize, learning_rate: f64, loss: f32) {
        self.samples_processed += samples;
        self.loss.push((self.samples_processed, loss));
        self.learning_rate
            .push((self.samples_processed, learning_rate));
    }

    /// Computes the average loss within the epoch.
    pub fn avg_loss(&self) -> f32 {
        let loss_sum = self.loss.iter().map(|(_, value)| *value).sum::<f32>();
        loss_sum / self.loss.len() as f32
    }

    /// Finalizes the epoch with an assessment matrix on the dev set.
    pub fn finalize<S>(
        self,
        samples: S,
        samples_classes: Tensor,
        errors: ThresholdInfo,
    ) -> EpochReport<S::Item>
    where
        S: IntoIterator,
        S::Item: AsRef<str>,
    {
        EpochReport {
            info: self,
            samples: samples.into_iter().collect(),
            samples_classes,
            errors,
        }
    }
}

/// A finalized training epoch report.
pub struct EpochReport<Sample> {
    info: EpochReportRecorder,
    samples: Vec<Sample>,
    samples_classes: Tensor,
    errors: ThresholdInfo,
}

/// Epoch data plot error.
#[derive(Debug, Snafu)]
#[snafu(context(false))]
#[cfg(feature = "plotters")]
pub struct PlotError {
    source: DrawAreaError,
}

#[cfg(feature = "plotters")]
type DrawAreaError = plotters::prelude::DrawingAreaErrorKind<
    <plotters::prelude::BitMapBackend<'static> as plotters::prelude::DrawingBackend>::ErrorType,
>;

#[cfg(feature = "plotters")]
impl<Sample> EpochReport<Sample> {
    /// Plot some graphs.
    pub fn plot<P>(&self, output: P) -> Result<(), PlotError>
    where
        P: AsRef<Utf8Path>,
    {
        use itertools::Itertools;
        use plotters::prelude::*;

        let output = output.as_ref();
        let root = BitMapBackend::new(output, (2048, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let x_range = match self.info.learning_rate.iter().map(|(x, _)| *x).minmax() {
            itertools::MinMaxResult::NoElements => {
                eprintln!("Nothing to plot");
                return Ok(());
            }
            itertools::MinMaxResult::OneElement(v) => 0f32..(v as f32) * 1.1,
            itertools::MinMaxResult::MinMax(min, max) => {
                let width = (max - min) as f32;
                min as f32..(max as f32 + width * 0.1)
            }
        };
        let loss_range = match self.info.loss.iter().map(|(_, value)| *value).minmax() {
            itertools::MinMaxResult::NoElements => {
                eprintln!("Nothing to plot");
                return Ok(());
            }
            itertools::MinMaxResult::OneElement(v) => 0f32..v * 1.1,
            itertools::MinMaxResult::MinMax(min, max) => {
                let width = max - min;
                (min - width * 0.05)..(max + width * 0.05)
            }
        };
        let lr_range = match self
            .info
            .learning_rate
            .iter()
            .map(|(_, value)| *value)
            .minmax()
        {
            itertools::MinMaxResult::NoElements => {
                eprintln!("Nothing to plot");
                return Ok(());
            }
            itertools::MinMaxResult::OneElement(v) => 0f64..v * 1.1,
            itertools::MinMaxResult::MinMax(min, max) => {
                let width = max - min;
                (min - width * 0.05)..(max + width * 0.05)
            }
        };

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .right_y_label_area_size(50)
            .margin(5)
            .caption("Learning rate and loss", ("sans-serif", 50.0).into_font())
            .build_cartesian_2d(x_range.clone(), loss_range)?
            .set_secondary_coord(x_range, lr_range);

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .y_desc("Loss")
            .y_label_formatter(&|x| format!("{:.2e}", x))
            .draw()?;

        chart
            .configure_secondary_axes()
            .y_desc("LR")
            .y_label_formatter(&|x| format!("{:.2e}", x))
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                self.info.loss.iter().map(|(x, y)| (*x as f32, *y)),
                &BLUE,
            ))?
            .label("y = loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart
            .draw_secondary_series(LineSeries::new(
                self.info.learning_rate.iter().map(|(x, y)| (*x as f32, *y)),
                &RED,
            ))?
            .label("y = lr")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart
            .configure_series_labels()
            .background_style(RGBColor(128, 128, 128))
            .draw()?;

        // To avoid the IO failure being ignored silently, we manually call the present function
        root.present()?; //.expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

        Ok(())
    }
}

/// Save error.
#[derive(Debug, Snafu)]
pub enum SaveError {
    /// Unable to create a report file.
    #[snafu(display("Unable to create a report file"))]
    CreateFile {
        /// Source error.
        source: std::io::Error,
    },
    /// Unable to save a report file.
    #[snafu(display("Unable to save a report file"))]
    SaveFile {
        /// Source error.
        source: std::io::Error,
    },
}

impl<Sample> EpochReport<Sample>
where
    Sample: AsRef<str>,
{
    /// Saves the info to the file.
    pub fn save<P>(&self, path: P) -> Result<(), SaveError>
    where
        P: AsRef<Utf8Path>,
    {
        use std::io::Write as _;
        let path = path.as_ref();
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).context(CreateFileSnafu)?);
        writeln!(f, "{self}").context(SaveFileSnafu)
    }
}

impl<Sample> std::fmt::Display for EpochReport<Sample>
where
    Sample: AsRef<str>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // let (best_threshold, best_f1) = self.errors.best(1.);

        writeln!(
            f,
            "Epoch #{}: samples count = {}, \
             average loss = {}",
            self.info.epoch,
            self.info.samples_count,
            self.info.avg_loss(),
        )?;

        for beta in [0.5, 1.0, 2.0] {
            let (threshold, score) = self.errors.best(beta);
            writeln!(
                f,
                "Best F_{beta} score: {}% at t = {}",
                score * 100.,
                threshold as f64 / 100.
            )?;
        }
        let (best_threshold, _) = self.errors.best(0.5);

        // Precision and recall.
        writeln!(f, "{}\n", self.errors)?;

        writeln!(f, "Samples (t = {}):", best_threshold as f64 / 100.)?;
        for (sentence, predicted) in self.samples.iter().zip(self.samples_classes.split(1, 0)) {
            let sentence = sentence.as_ref();
            let predicted: f32 = predicted.try_into().expect("Must be float");
            let is_spam = (predicted * 100.).round() as u64 >= best_threshold;
            let color = if is_spam {
                AnsiColors::Red
            } else {
                AnsiColors::Green
            };
            writeln!(
                f,
                "spam: {}%. {sentence}",
                format_args!("{: >6.2}", predicted * 100.).color(color)
            )?;
        }
        Ok(())
    }
}
