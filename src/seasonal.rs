use crate::types::*;

/// Seasonal decomposition using moving averages. Returns (trend, seasonal, residual).
pub fn seasonal_decompose(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    if period < 2 || n < period * 2 {
        let zeros = vec![0.0; n];
        return (data.to_vec(), zeros.clone(), zeros);
    }

    // Compute trend using centered moving average with the period
    let mut trend = vec![f64::NAN; n];
    let half = period / 2;
    for i in half..n - half {
        let start = i - half;
        let end = (i + half + 1).min(n);
        trend[i] = data[start..end].iter().sum::<f64>() / (end - start) as f64;
    }
    // Fill edges with nearest valid trend value
    let first_valid = trend.iter().position(|t| !t.is_nan()).unwrap_or(n - 1);
    let last_valid = trend.iter().rposition(|t| !t.is_nan()).unwrap_or(0);
    for i in 0..first_valid {
        trend[i] = trend[first_valid];
    }
    for i in (last_valid + 1)..n {
        trend[i] = trend[last_valid];
    }

    // Detrend: data - trend
    let detrended: Vec<f64> = data.iter().zip(trend.iter()).map(|(d, t)| d - t).collect();

    // Compute seasonal component: average detrended value for each phase
    let mut seasonal = vec![0.0; n];
    let mut counts = vec![0usize; period];
    let mut sums = vec![0.0f64; period];
    for (i, &d) in detrended.iter().enumerate() {
        let phase = i % period;
        sums[phase] += d;
        counts[phase] += 1;
    }
    for phase in 0..period {
        if counts[phase] > 0 {
            sums[phase] /= counts[phase] as f64;
        }
    }
    // Center seasonal component (mean = 0)
    let seasonal_mean = sums.iter().sum::<f64>() / period as f64;
    for phase in 0..period {
        sums[phase] -= seasonal_mean;
    }
    for i in 0..n {
        seasonal[i] = sums[i % period];
    }

    // Residual = data - trend - seasonal
    let residual: Vec<f64> = data
        .iter()
        .zip(trend.iter().zip(seasonal.iter()))
        .map(|(d, (t, s))| d - t - s)
        .collect();

    (trend, seasonal, residual)
}

/// Detect anomalies using seasonal decomposition. Flags points with large residuals.
pub fn seasonal_detect(data: &[f64], period: usize, threshold: f64) -> DetectionResult {
    if data.is_empty() {
        return DetectionResult::new(vec![], threshold, 0);
    }
    let (_trend, _seasonal, residual) = seasonal_decompose(data, period);
    let n = residual.len() as f64;
    let res_mean = residual.iter().sum::<f64>() / n;
    let res_std = {
        let v = residual.iter().map(|r| (r - res_mean).powi(2)).sum::<f64>() / n;
        v.sqrt()
    };
    if res_std < 1e-10 {
        return DetectionResult::new(vec![], threshold, data.len());
    }

    let anomalies: Vec<Anomaly> = residual
        .iter()
        .enumerate()
        .filter_map(|(i, &r)| {
            let z = (r - res_mean).abs() / res_std;
            if z > threshold {
                Some(Anomaly {
                    index: i,
                    value: data[i],
                    score: z,
                    method: "seasonal".into(),
                    context: vec![res_mean, res_std, period as f64],
                })
            } else {
                None
            }
        })
        .collect();
    DetectionResult::new(anomalies, threshold, data.len())
}
