use serde::{Deserialize, Serialize};

/// A single detected anomaly point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub index: usize,
    pub value: f64,
    pub score: f64,
    pub method: String,
    pub context: Vec<f64>,
}

/// Available anomaly detection methods.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyDetector {
    ZScore,
    IQR,
    IsolationForest,
    Spectral,
    Seasonal,
    ModifiedZScore,
    MovingZScore,
    Grubbs,
    Contextual,
}

/// Result of an anomaly detection run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub anomalies: Vec<Anomaly>,
    pub threshold: f64,
    pub total_points: usize,
    pub anomaly_rate: f64,
}

/// Configuration for anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub method: AnomalyDetector,
    pub threshold: f64,
    pub window: usize,
    pub seasonal_period: Option<usize>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            method: AnomalyDetector::ZScore,
            threshold: 3.0,
            window: 20,
            seasonal_period: None,
        }
    }
}

impl DetectionResult {
    pub(crate) fn new(anomalies: Vec<Anomaly>, threshold: f64, total_points: usize) -> Self {
        let anomaly_rate = if total_points > 0 {
            anomalies.len() as f64 / total_points as f64
        } else {
            0.0
        };
        Self {
            anomalies,
            threshold,
            total_points,
            anomaly_rate,
        }
    }
}
