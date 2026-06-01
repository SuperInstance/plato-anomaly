# plato-anomaly

> Anomaly detection for PLATO tile streams — Z-score, spectral, seasonal, and ensemble methods

## What This Does

plato-anomaly provides multiple anomaly detection algorithms for tile data: Z-score (statistical outliers), spectral (frequency-domain anomalies), seasonal (deviations from expected seasonal patterns), and ensemble (combining multiple detectors). Each detector produces anomaly scores with configurable thresholds.

## The Key Idea

An anomaly is a data point that doesn't belong. But "doesn't belong" depends on context. A temperature of 30°C is normal in summer but anomalous in winter. plato-anomaly provides multiple lenses: Z-score catches statistical outliers, spectral analysis catches frequency-domain oddities, seasonal detection catches deviations from periodic patterns, and ensemble combines them all for robustness.

## Install

```bash
cargo add plato-anomaly
```

## Quick Start

```rust
use plato_anomaly::*;

let data = vec![20.0, 21.0, 20.5, 22.0, 35.0, 21.0, 20.0]; // 35.0 is anomalous

// Z-score detection
let anomalies = detect_zscore(&data, 2.0); // 2 std devs
// anomalies contains the index of 35.0

// Ensemble: combine multiple detectors
let ensemble = EnsembleDetector::new()
    .with_zscore(2.0)
    .with_spectral(0.3)
    .with_seasonal(24); // 24-hour period
let results = ensemble.detect(&data);
```

## API Reference

### Detection Modules

| Module | Description |
|---|---|
| `detect` | Z-score based anomaly detection |
| `spectral` | Frequency-domain anomaly detection via DFT |
| `seasonal` | Seasonal decomposition and deviation detection |
| `ensemble` | Combine multiple detectors |

### Types

| Type | Description |
|---|---|
| `AnomalyScore { value, score, threshold, is_anomaly }` | Scored observation |
| `EnsembleDetector` | Configurable combination of detectors |

## Testing

23 tests: Z-score detection, spectral analysis, seasonal decomposition, ensemble voting, threshold sensitivity, edge cases.

## License

Apache-2.0
