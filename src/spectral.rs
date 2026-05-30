use crate::types::*;

/// FFT-based spectral anomaly detection. Flags points where energy appears in unexpected frequencies.
pub fn spectral_detect(data: &[f64], threshold: f64) -> DetectionResult {
    if data.len() < 4 {
        return DetectionResult::new(vec![], threshold, data.len());
    }

    // Compute DFT magnitudes (we implement our own simple DFT for zero dependencies)
    let n = data.len();
    let magnitudes = compute_dft_magnitudes(data);
    let total_energy: f64 = magnitudes.iter().sum();
    if total_energy == 0.0 {
        return DetectionResult::new(vec![], threshold, n);
    }

    // Compute energy contribution of each point
    let mean = data.iter().sum::<f64>() / n as f64;
    let std_dev = {
        let v = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        v.sqrt()
    };

    // Use frequency-domain features to score each point
    // We reconstruct from dominant frequencies and check residuals
    let num_freqs = magnitudes.len();
    let max_mag = magnitudes.iter().cloned().fold(0.0f64, f64::max);
    if max_mag == 0.0 {
        return DetectionResult::new(vec![], threshold, n);
    }

    // Normalize magnitudes
    let norm_mags: Vec<f64> = magnitudes.iter().map(|m| m / max_mag).collect();

    // For each point, compute its contribution to high-frequency energy
    let mut scores = vec![0.0f64; n];
    for (k, &mag) in norm_mags.iter().enumerate().skip(1) {
        // High frequencies get more weight
        let freq_weight = k as f64 / num_freqs as f64;
        for t in 0..n {
            let phase = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            scores[t] += mag * freq_weight * phase.cos().abs();
        }
    }

    // Normalize scores
    let score_mean = scores.iter().sum::<f64>() / n as f64;
    let score_std = {
        let v = scores.iter().map(|s| (s - score_mean).powi(2)).sum::<f64>() / n as f64;
        v.sqrt()
    };

    let anomalies: Vec<Anomaly> = if score_std > 1e-10 {
        scores
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| {
                let z = (s - score_mean) / score_std;
                if z > threshold {
                    Some(Anomaly {
                        index: i,
                        value: data[i],
                        score: z,
                        method: "spectral".into(),
                        context: vec![mean, std_dev],
                    })
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![]
    };

    DetectionResult::new(anomalies, threshold, n)
}

/// Compute DFT magnitude spectrum (only positive frequencies, DC excluded).
fn compute_dft_magnitudes(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let half = n / 2 + 1;
    let mut magnitudes = Vec::with_capacity(half);

    for k in 0..half {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, &x) in data.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt() / n as f64);
    }
    magnitudes
}
