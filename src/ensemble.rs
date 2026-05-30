use crate::types::*;
use crate::{iqr_detect, modified_zscore_detect, moving_zscore_detect, spectral_detect, zscore_detect};

/// Ensemble anomaly detection: majority vote across multiple methods.
pub fn ensemble_detect(data: &[f64], methods: &[AnomalyDetector], threshold: f64) -> DetectionResult {
    if data.is_empty() {
        return DetectionResult::new(vec![], threshold, 0);
    }
    let n = data.len();
    let mut votes = vec![0usize; n];
    let method_count = methods.len();

    for method in methods {
        let result = match method {
            AnomalyDetector::ZScore => zscore_detect(data, threshold),
            AnomalyDetector::IQR => iqr_detect(data, 1.5),
            AnomalyDetector::ModifiedZScore => modified_zscore_detect(data, threshold),
            AnomalyDetector::MovingZScore => moving_zscore_detect(data, 20.min(n / 2).max(3), threshold),
            AnomalyDetector::Spectral => spectral_detect(data, threshold),
            AnomalyDetector::Seasonal => {
                if n >= 24 {
                    crate::seasonal_detect(data, 12.min(n / 2), threshold)
                } else {
                    zscore_detect(data, threshold)
                }
            }
            _ => zscore_detect(data, threshold),
        };
        for anomaly in &result.anomalies {
            if anomaly.index < n {
                votes[anomaly.index] += 1;
            }
        }
    }

    // Majority: more than half the methods must agree
    let required_votes = (method_count / 2).max(1);
    let anomalies: Vec<Anomaly> = (0..n)
        .filter(|&i| votes[i] >= required_votes)
        .map(|i| Anomaly {
            index: i,
            value: data[i],
            score: votes[i] as f64 / method_count as f64,
            method: "ensemble".into(),
            context: vec![votes[i] as f64, method_count as f64],
        })
        .collect();

    DetectionResult::new(anomalies, threshold, n)
}
