use crate::types::*;

/// Detect anomalies using z-score: points beyond `threshold` standard deviations from the mean.
pub fn zscore_detect(data: &[f64], threshold: f64) -> DetectionResult {
    if data.is_empty() {
        return DetectionResult::new(vec![], threshold, 0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std_dev = {
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    };
    if std_dev == 0.0 {
        return DetectionResult::new(vec![], threshold, data.len());
    }
    let anomalies: Vec<Anomaly> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| {
            let z = (v - mean).abs() / std_dev;
            z > threshold
        })
        .map(|(i, &v)| {
            let z = (v - mean).abs() / std_dev;
            Anomaly {
                index: i,
                value: v,
                score: z,
                method: "zscore".into(),
                context: vec![mean, std_dev],
            }
        })
        .collect();
    DetectionResult::new(anomalies, threshold, data.len())
}

/// Detect anomalies using IQR: points outside Q1 - multiplier*IQR to Q3 + multiplier*IQR.
pub fn iqr_detect(data: &[f64], multiplier: f64) -> DetectionResult {
    if data.is_empty() {
        return DetectionResult::new(vec![], multiplier, 0);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[(3 * n) / 4];
    let iqr = q3 - q1;
    let lower = q1 - multiplier * iqr;
    let upper = q3 + multiplier * iqr;

    let anomalies: Vec<Anomaly> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v < lower || v > upper)
        .map(|(i, &v)| {
            let score = if v < lower {
                (lower - v) / (iqr + 1e-10)
            } else {
                (v - upper) / (iqr + 1e-10)
            };
            Anomaly {
                index: i,
                value: v,
                score,
                method: "iqr".into(),
                context: vec![q1, q3, iqr],
            }
        })
        .collect();
    DetectionResult::new(anomalies, multiplier, data.len())
}

/// Modified z-score using Median Absolute Deviation (MAD). More robust for heavy-tailed distributions.
pub fn modified_zscore_detect(data: &[f64], threshold: f64) -> DetectionResult {
    if data.is_empty() {
        return DetectionResult::new(vec![], threshold, 0);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mut deviations: Vec<f64> = data.iter().map(|x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = deviations[deviations.len() / 2];
    if mad == 0.0 {
        return DetectionResult::new(vec![], threshold, data.len());
    }
    // 0.6745 is the 75th percentile of the standard normal distribution
    let anomalies: Vec<Anomaly> = data
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            let modified_z = 0.6745 * (v - median) / mad;
            if modified_z.abs() > threshold {
                Some(Anomaly {
                    index: i,
                    value: v,
                    score: modified_z.abs(),
                    method: "modified_zscore".into(),
                    context: vec![median, mad],
                })
            } else {
                None
            }
        })
        .collect();
    DetectionResult::new(anomalies, threshold, data.len())
}

/// Grubbs' test for a single outlier. Returns the index of the most extreme value if significant.
pub fn grubbs_test(data: &[f64], alpha: f64) -> Option<usize> {
    if data.len() < 3 {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std_dev = {
        let v = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        v.sqrt()
    };
    if std_dev == 0.0 {
        return None;
    }
    let (max_idx, &max_dev) = data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            (*a - mean).abs().partial_cmp(&(*b - mean).abs()).unwrap()
        })
        .unwrap();
    let g = (max_dev - mean).abs() / std_dev;
    // Critical value approximation using t-distribution
    let t_crit = approx_t_critical(alpha / (2.0 * n), n - 2.0);
    let g_crit = ((n - 1.0) / n.sqrt()) * (t_crit / ((n - 2.0 + t_crit * t_crit).sqrt()));

    if g > g_crit {
        Some(max_idx)
    } else {
        None
    }
}

/// Approximate t critical value using a simple inverse for common alpha levels.
fn approx_t_critical(alpha: f64, _df: f64) -> f64 {
    // Simplified: use normal approximation for large-ish samples
    // For a proper implementation you'd use a stats library
    if alpha <= 0.001 {
        3.291
    } else if alpha <= 0.01 {
        2.576
    } else if alpha <= 0.025 {
        2.241
    } else if alpha <= 0.05 {
        1.96
    } else if alpha <= 0.1 {
        1.645
    } else {
        1.282
    }
}

/// Moving (rolling) z-score anomaly detection. Detects local anomalies in data with shifting baseline.
pub fn moving_zscore_detect(data: &[f64], window: usize, threshold: f64) -> DetectionResult {
    if data.len() < window || window < 2 {
        return DetectionResult::new(vec![], threshold, data.len());
    }
    let mut anomalies = Vec::new();
    let half = window / 2;

    for i in half..data.len() - half {
        let start = i.saturating_sub(half);
        let end = (i + half).min(data.len());
        let local: Vec<f64> = data[start..end].to_vec();
        let local_n = local.len() as f64;
        let local_mean = local.iter().sum::<f64>() / local_n;
        let local_std = {
            let v = local.iter().map(|x| (x - local_mean).powi(2)).sum::<f64>() / local_n;
            v.sqrt()
        };
        if local_std < 1e-10 {
            continue;
        }
        let z = (data[i] - local_mean).abs() / local_std;
        if z > threshold {
            anomalies.push(Anomaly {
                index: i,
                value: data[i],
                score: z,
                method: "moving_zscore".into(),
                context: vec![local_mean, local_std, window as f64],
            });
        }
    }
    DetectionResult::new(anomalies, threshold, data.len())
}

/// Contextual anomaly detection: value is normal globally but anomalous given a context signal.
pub fn contextual_detect(data: &[f64], context: &[f64], threshold: f64) -> DetectionResult {
    if data.is_empty() || data.len() != context.len() {
        return DetectionResult::new(vec![], threshold, data.len());
    }
    let n = data.len() as f64;
    // Compute residuals from a simple linear relationship
    let ctx_mean = context.iter().sum::<f64>() / n;
    let data_mean = data.iter().sum::<f64>() / n;
    let cov: f64 = data.iter().zip(context.iter()).map(|(d, c)| (d - data_mean) * (c - ctx_mean)).sum::<f64>() / n;
    let ctx_var: f64 = context.iter().map(|c| (c - ctx_mean).powi(2)).sum::<f64>() / n;
    if ctx_var < 1e-10 {
        return DetectionResult::new(vec![], threshold, data.len());
    }
    let slope = cov / ctx_var;
    let intercept = data_mean - slope * ctx_mean;

    let residuals: Vec<f64> = data
        .iter()
        .zip(context.iter())
        .map(|(d, c)| d - (intercept + slope * c))
        .collect();

    let res_mean = residuals.iter().sum::<f64>() / n;
    let res_std = {
        let v = residuals.iter().map(|r| (r - res_mean).powi(2)).sum::<f64>() / n;
        v.sqrt()
    };
    if res_std < 1e-10 {
        return DetectionResult::new(vec![], threshold, data.len());
    }

    let anomalies: Vec<Anomaly> = residuals
        .iter()
        .enumerate()
        .filter_map(|(i, &r)| {
            let z = (r - res_mean).abs() / res_std;
            if z > threshold {
                Some(Anomaly {
                    index: i,
                    value: data[i],
                    score: z,
                    method: "contextual".into(),
                    context: vec![context[i], intercept, slope],
                })
            } else {
                None
            }
        })
        .collect();
    DetectionResult::new(anomalies, threshold, data.len())
}
