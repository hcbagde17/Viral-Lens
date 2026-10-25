# 📋 Viral-Lens: Phase 2 Data Quality Report & Audit Findings

## 1. Executive Summary

- **Training Set (`data/raw/train.csv`)**: 7,000 rows $\times$ 8 columns (`video_id`, `title`, `thumbnail_url`, `published_at`, `duration`, `view_count`, `subscriber_count`, `virality_score`).
- **Test Set (`data/raw/test.csv`)**: 3,014 rows $\times$ 6 columns (`video_id`, `title`, `thumbnail_url`, `published_at`, `duration`, `subscriber_count`).
- **Integrity**:
  - Null counts: **0** across all columns in both train and test.
  - Duplicate `video_id`s: **0** in train, **0** in test.
  - Cross-split overlap: **0** shared `video_id`s between train and test.

---

## 2. Virality Score Formula Verification

The domain formulation defines:
$$\text{virality\_score} \approx \ln(\text{view\_count} + 1) - \ln(\text{subscriber\_count} + 1)$$

- **Maximum absolute difference**: $1.33 \times 10^{-15}$
- **Mean absolute difference**: $5.72 \times 10^{-17}$
- **Rows with $|\Delta| > 10^{-4}$**: **0** (100% agreement to IEEE 754 64-bit float precision).
- **Decision**: All training labels are mathematically consistent; no rows need to be dropped due to label corruption.

---

## 3. Discovered Anomalies, Edge Cases & Data Quality Issues

### Issue 1: Zero Video Duration (`duration == 0`)
- **Observation**: 38 rows in train (0.54%) and 12 rows in test (0.40%) have `duration == 0`.
- **Root Cause**: YouTube metadata occasionally assigns duration 0 to live stream archives, premiere announcements, or during initial indexing.
- **Serving / Test Impact**: Test rows cannot be dropped because an inference prediction is mandatory for every test `video_id`.
- **Resolution**:
  - Feature transformation $\ln(\text{duration} + 1)$ is well-defined: $\ln(0 + 1) = 0.0$.
  - Binary flag `is_short = int(duration <= 60)` evaluates to `1` for these rows.
  - In `core/predictor.py` and API validation, clamp or permit $\text{duration} \ge 0$ (or clamp `duration = max(1, duration)` if $\ge 1$ validation is enforced).

### Issue 2: Extreme Distribution Skewness
- **Observation**:
  - `view_count`: Raw skewness = **15.46** (range: 1,005 to 2,286,773,000; median: 2.86M).
  - `subscriber_count`: Raw skewness = **14.50** (range: 62 to 457,000,000; median: 516K).
  - `duration`: Raw skewness = **13.44** (range: 0 to 219,013s / ~60.8 hours; median: 52s).
- **Resolution**:
  - Natural logarithmic transforms $\ln(x + 1)$ reduce skewness to near-Gaussian:
    - $\ln(\text{view\_count} + 1)$: skew = **-0.55**
    - $\ln(\text{subscriber\_count} + 1)$: skew = **-0.33**
    - $\ln(\text{duration} + 1)$: skew = **1.09**
  - Standardizing $\ln(\text{subscriber\_count} + 1)$ and $\ln(\text{duration} + 1)$ using training-only statistics avoids gradient explosion in the neural network.

### Issue 3: High Prevalence of YouTube Shorts
- **Observation**: 62.89% of train videos and 62.67% of test videos have $\text{duration} \le 60\text{ seconds}$.
- **Resolution**: The tabular branch will include an explicit indicator `is_short = 1` for duration $\le 60$s, allowing the model to learn distinct virality baselines for Shorts vs long-form video.

### Issue 4: Multilingual Titles & Emojis
- **Observation**:
  - **44.33%** of train titles and **45.06%** of test titles contain emojis or non-ASCII characters (e.g., Devanagari, Japanese Kanji, Cyrillic, accented Latin).
  - Character lengths: min = 2, median = 63, max = 100 (YouTube strict cap).
  - Word counts: min = 1, median = 10, max = 26.
- **Resolution**:
  - Use `distilbert-base-multilingual-cased` to preserve multilingual subwords and emojis rather than stripping them.
  - Max token sequence length set to **32 tokens**, which covers $>99\%$ of titles without truncation.

### Issue 5: Thumbnail Source Uniformity
- **Observation**: 100% of thumbnail URLs point to Google CDN (`i.ytimg.com`).
- **Resolution**: Handled in Phase 3 with concurrent retries, Pillow format verification, and a minimum width threshold ($\ge 200\text{px}$) to filter out YouTube gray default placeholders.

### Issue 6: Channel Leakage Risk
- **Observation**: The raw dataset omits channel IDs, but videos from the same creator have identical `subscriber_count` values (2,445 unique subscriber tiers across 7,000 rows).
- **Resolution**: Offline validation must use `GroupShuffleSplit` grouped by `subscriber_count` to ensure a creator's videos do not appear in both train and validation sets.

---

## 4. Cleaning Decisions & Record

| Rule | Train Action | Test Action | Justification |
| :--- | :--- | :--- | :--- |
| Null / negative `view_count` | 0 rows found | N/A (no target) | Data is already clean |
| Null `subscriber_count` | 0 rows found | 0 rows found | Data is already clean |
| Empty / whitespace title | 0 rows found | 0 rows found | Data is already clean |
| Disagreeing virality score | 0 rows found | N/A | Formula verified to $1.33 \times 10^{-15}$ |
| `duration == 0` | Retained (38 rows) | Retained (12 rows) | Handled safely via $\ln(\text{duration}+1) = 0.0$; test rows cannot be dropped |

**Total Rows Retained**:
- `train.csv`: 7,000 / 7,000 (100%)
- `test.csv`: 3,014 / 3,014 (100%)
