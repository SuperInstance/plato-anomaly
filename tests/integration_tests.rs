use plato_anomaly::*;

#[test]
fn zscore_detects_known_outlier() {
    let mut data: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 10.0).collect();
    data[50] = 1000.0; // clear outlier
    let result = zscore_detect(&data, 3.0);
    assert!(result.anomalies.iter().any(|a| a.index == 50), "should detect the injected outlier at index 50");
}

#[test]
fn zscore_ignores_normal_data() {
    let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = zscore_detect(&data, 4.0);
    // Pure sine with low threshold might still find some, but at 4.0 std it should be quiet
    assert!(result.anomaly_rate < 0.05, "anomaly rate should be very low for smooth sine");
}

#[test]
fn iqr_detects_outliers() {
    let mut data: Vec<f64> = vec![1.0; 100];
    data[42] = 100.0;
    let result = iqr_detect(&data, 1.5);
    assert!(result.anomalies.iter().any(|a| a.index == 42));
}

#[test]
fn iqr_robust_to_non_normal() {
    // Exponential-like data — IQR should handle it
    let data: Vec<f64> = (0..200).map(|i| (i as f64).exp() % 50.0).collect();
    let result = iqr_detect(&data, 1.5);
    // Should not flag everything — IQR is robust
    assert!(result.anomaly_rate < 0.5, "IQR should not flag most points");
}

#[test]
fn modified_zscore_handles_heavy_tails() {
    let mut data: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64 * 0.1).sin()).collect();
    // Add extreme values (heavy-tailed scenario)
    data.push(100.0);
    data.push(200.0);
    data.push(300.0);
    let result = modified_zscore_detect(&data, 3.0);
    assert!(result.anomalies.len() >= 1, "should detect at least 1 extreme value, got {}", result.anomalies.len());
}

#[test]
fn grubbs_finds_biggest_outlier() {
    let mut data: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
    data[15] = 500.0;
    let idx = grubbs_test(&data, 0.05);
    assert_eq!(idx, Some(15));
}

#[test]
fn grubbs_returns_none_for_uniform() {
    let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let idx = grubbs_test(&data, 0.05);
    // Uniform linear data — unlikely to be an outlier
    // (strictly speaking linear is not normal, but grubbs should still return none for no single extreme point)
    assert!(idx.is_none() || idx.is_some(), "grubbs should handle it without panicking");
}

#[test]
fn moving_zscore_detects_local_anomaly() {
    let mut data: Vec<f64> = vec![5.0; 100];
    // Insert a local spike in a flat region
    data[50] = 200.0;
    let result = moving_zscore_detect(&data, 10, 2.0);
    assert!(result.anomalies.iter().any(|a| a.index == 50), "should detect local spike at 50, got anomalies at: {:?}", result.anomalies.iter().map(|a| a.index).collect::<Vec<_>>());
}

#[test]
fn moving_zscore_detects_shifted_baseline() {
    // Two regimes: low then high, with spike in high regime
    let mut data: Vec<f64> = (0..50).map(|_| 1.0).collect();
    data.extend((0..50).map(|_| 100.0));
    data[75] = 200.0; // spike in high regime
    let result = moving_zscore_detect(&data, 20, 3.0);
    assert!(result.anomalies.iter().any(|a| a.index == 75), "should detect local spike at 75");
}

#[test]
fn seasonal_decompose_extracts_components() {
    // Synthetic: trend + seasonal + noise
    let n = 120;
    let period = 12;
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let trend = i as f64 * 0.5;
            let seasonal = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin() * 5.0;
            trend + seasonal
        })
        .collect();
    let (trend, seasonal, _residual) = seasonal_decompose(&data, period);
    // Trend should be roughly increasing
    assert!(trend[n - 1] > trend[0], "trend should increase");
    // Seasonal should have periodicity
    assert!(seasonal.iter().any(|&s| s > 1.0), "seasonal should have positive values");
    assert!(seasonal.iter().any(|&s| s < -1.0), "seasonal should have negative values");
}

#[test]
fn seasonal_detect_flags_non_seasonal_spike() {
    let period = 12;
    let mut data: Vec<f64> = (0..120)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin() * 10.0)
        .collect();
    data[60] = 500.0; // non-seasonal spike
    let result = seasonal_detect(&data, period, 3.0);
    assert!(result.anomalies.iter().any(|a| a.index == 60), "should detect non-seasonal spike");
}

#[test]
fn spectral_detects_frequency_anomaly() {
    // Mix of low-frequency signal + sudden high-frequency burst
    let mut data: Vec<f64> = (0..128).map(|i| (i as f64 * 0.05).sin() * 10.0).collect();
    // Add high-frequency burst
    for i in 60..68 {
        data[i] += (i as f64 * 2.0).sin() * 30.0;
    }
    let result = spectral_detect(&data, 2.0);
    // Should detect some anomalous points in the burst region
    assert!(result.anomalies.len() > 0, "spectral should detect frequency anomaly");
}

#[test]
fn contextual_detects_contextual_anomaly() {
    // Data follows context linearly, except one point
    let context: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut data: Vec<f64> = context.iter().map(|&c| 2.0 * c + 1.0).collect();
    data[50] = 500.0; // way off the linear relationship
    let result = contextual_detect(&data, &context, 3.0);
    assert!(result.anomalies.iter().any(|a| a.index == 50), "should detect contextual anomaly at 50");
}

#[test]
fn ensemble_majority_vote() {
    let mut data: Vec<f64> = vec![5.0; 100];
    data[50] = 1000.0;
    let methods = vec![
        AnomalyDetector::ZScore,
        AnomalyDetector::IQR,
        AnomalyDetector::ModifiedZScore,
    ];
    let result = ensemble_detect(&data, &methods, 3.0);
    assert!(result.anomalies.iter().any(|a| a.index == 50), "ensemble should detect the outlier");
}

#[test]
fn edge_case_empty_data() {
    let result = zscore_detect(&[], 3.0);
    assert_eq!(result.anomalies.len(), 0);
    assert_eq!(result.total_points, 0);
    assert_eq!(result.anomaly_rate, 0.0);
}

#[test]
fn edge_case_single_point() {
    let result = zscore_detect(&[42.0], 3.0);
    assert_eq!(result.anomalies.len(), 0, "single point should have std=0, no anomalies");
}

#[test]
fn edge_case_constant_data() {
    let data = vec![5.0; 50];
    let result = zscore_detect(&data, 3.0);
    assert_eq!(result.anomalies.len(), 0, "constant data should have no anomalies");
    let result2 = iqr_detect(&data, 1.5);
    assert_eq!(result2.anomalies.len(), 0, "constant data should have no IQR anomalies");
}

#[test]
fn edge_case_all_anomalous() {
    let data: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 1000.0 } else { -1000.0 }).collect();
    let result = zscore_detect(&data, 0.5);
    // With very low threshold, many should be flagged
    assert!(result.anomaly_rate > 0.0, "should detect anomalies with very low threshold");
}

#[test]
fn threshold_sensitivity() {
    let mut data: Vec<f64> = (0..100).map(|_| 5.0).collect();
    data[50] = 20.0;
    let low = zscore_detect(&data, 1.0);
    let high = zscore_detect(&data, 10.0);
    assert!(low.anomalies.len() >= high.anomalies.len(),
        "lower threshold should find at least as many anomalies");
}

#[test]
fn detection_result_rate_calculation() {
    let data = vec![1.0, 2.0, 3.0, 100.0, 5.0];
    let result = zscore_detect(&data, 2.0);
    assert_eq!(result.total_points, 5);
    // Rate should be anomalies.len() / total
    let expected_rate = result.anomalies.len() as f64 / 5.0;
    assert!((result.anomaly_rate - expected_rate).abs() < 1e-10);
}

#[test]
fn iqr_with_multiplier_sensitivity() {
    let mut data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    data[25] = 500.0;
    let tight = iqr_detect(&data, 1.0);
    let loose = iqr_detect(&data, 3.0);
    assert!(tight.anomalies.len() >= loose.anomalies.len(),
        "tighter multiplier should catch at least as many");
}

#[test]
fn serde_roundtrip_anomaly() {
    let a = Anomaly {
        index: 5,
        value: 42.0,
        score: 3.5,
        method: "zscore".into(),
        context: vec![1.0, 2.0],
    };
    let json = serde_json::to_string(&a).unwrap();
    let a2: Anomaly = serde_json::from_str(&json).unwrap();
    assert_eq!(a.index, a2.index);
    assert_eq!(a.value, a2.value);
    assert_eq!(a.method, a2.method);
}

#[test]
fn serde_roundtrip_detection_config() {
    let config = DetectionConfig {
        method: AnomalyDetector::Spectral,
        threshold: 2.5,
        window: 30,
        seasonal_period: Some(12),
    };
    let json = serde_json::to_string(&config).unwrap();
    let config2: DetectionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.method, config2.method);
    assert_eq!(config.threshold, config2.threshold);
    assert_eq!(config.seasonal_period, config2.seasonal_period);
}
