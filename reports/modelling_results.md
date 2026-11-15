# Modelling results

Recorded on 15 November 2026 from a complete run of the modelling analysis.
This is a dated snapshot. The notebook itself calculates fresh tables and explanations when rerun; this document does not update automatically.

**Model comparison**

The run uses 40,837 training customers. AP measures how well models rank churners; it is not classification accuracy. Higher is better.

| Model | Mean AP | Fold standard deviation |
| --- | --- | --- |
| Gradient Boosting (HistGB) | 0.4677 | 0.0120 |
| Random Forest (300 trees) | 0.4616 | 0.0076 |
| Logistic Regression | 0.4269 | 0.0098 |
| Decision Tree (depth 5) | 0.4066 | 0.0086 |
| Naive Bayes (Gaussian) | 0.3730 | 0.0074 |
| Fixed tenure (11-12 months) | 0.3133 | 0.0019 |
| Dummy (prior) | 0.2882 | 0.0001 |
| Dummy (stratified) | 0.2878 | 0.0026 |

The scores are development results. The same folds have been used for preprocessing and model choices. Fold standard deviations describe variation, not confidence intervals or significance. The reserved test set has not been evaluated.

**All eight group combinations**

Account, demographic and handset information stays in every combination. The columns below compare two model types with their settings kept fixed.

| Added groups | Gradient Boosting AP | Logistic Regression AP |
| --- | --- | --- |
| Background only | 0.4123 | 0.3937 |
| Usage | 0.4351 | 0.4123 |
| Usage change | 0.4321 | 0.4020 |
| Service quality | 0.4214 | 0.3979 |
| Usage + Usage change | 0.4651 | 0.4235 |
| Usage + Service quality | 0.4383 | 0.4165 |
| Usage change + Service quality | 0.4374 | 0.4056 |
| All three behavioural groups | 0.4677 | 0.4269 |

**What each group contributes**

“Added alone” compares background-plus-one-group with background only. “Removed from full” is the AP loss when removing that group and retraining. “Average over orders” averages the group's AP improvement over all six possible orders. The last column counts folds where this average is positive.

| Model | Group | Added alone | Removed from full | Average over orders | Folds with positive average |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting | usage | +0.0228 | +0.0303 | +0.0260 | 5/5 |
| Gradient Boosting | usage change | +0.0198 | +0.0294 | +0.0241 | 5/5 |
| Gradient Boosting | service quality | +0.0091 | +0.0026 | +0.0053 | 5/5 |
| Logistic Regression | usage | +0.0187 | +0.0213 | +0.0200 | 5/5 |
| Logistic Regression | usage change | +0.0084 | +0.0104 | +0.0094 | 5/5 |
| Logistic Regression | service quality | +0.0042 | +0.0034 | +0.0038 | 5/5 |

The largest average contribution is **usage** for Gradient Boosting and **usage** for Logistic Regression. These are rankings for these models, this background and this metric. Small differences provide limited evidence for a stable ranking.

The three average contributions sum to full-model AP minus background AP on each fold. This is a Shapley allocation of the measured improvement. Calculated features enter only when their parent groups are present, so their value is included and can be shared between their parent groups. This does not rank the three fixed background groups or establish cause and effect.

**Gradient Boosting: gains in each addition order**

| Addition order | Usage gain | Usage-change gain | Service-quality gain |
| --- | --- | --- | --- |
| Usage → Usage change → Service quality | +0.0228 | +0.0300 | +0.0026 |
| Usage → Service quality → Usage change | +0.0228 | +0.0294 | +0.0032 |
| Usage change → Usage → Service quality | +0.0330 | +0.0198 | +0.0026 |
| Usage change → Service quality → Usage | +0.0303 | +0.0198 | +0.0053 |
| Service quality → Usage → Usage change | +0.0169 | +0.0294 | +0.0091 |
| Service quality → Usage change → Usage | +0.0303 | +0.0160 | +0.0091 |

**Logistic Regression: gains in each addition order**

| Addition order | Usage gain | Usage-change gain | Service-quality gain |
| --- | --- | --- | --- |
| Usage → Usage change → Service quality | +0.0187 | +0.0112 | +0.0034 |
| Usage → Service quality → Usage change | +0.0187 | +0.0104 | +0.0042 |
| Usage change → Usage → Service quality | +0.0215 | +0.0084 | +0.0034 |
| Usage change → Service quality → Usage | +0.0213 | +0.0084 | +0.0035 |
| Service quality → Usage → Usage change | +0.0186 | +0.0104 | +0.0042 |
| Service quality → Usage change → Usage | +0.0213 | +0.0077 | +0.0042 |

All orders reuse the same eight fitted combinations. The order averages are calculated from these scores without further model fitting.

**Customer-segment analysis**

The model selected for this development analysis is **Gradient Boosting (HistGB)**, using the highest mean AP among the five learned models. The F1 threshold is **0.2429**. It flags **25,139 customers (61.6%)**, with precision **0.370**, recall **0.790**, F1 **0.504**, and **2,475 missed churners**.

| Tenure | Customers | Churners | Missed churners | Miss rate | Share of churners | Share of misses |
| --- | --- | --- | --- | --- | --- | --- |
| up to 10 months | 8,510 | 1,430 | 963 | 67.3% | 12.2% | 38.9% |
| 11-12 months | 5,090 | 2,161 | 19 | 0.9% | 18.4% | 0.8% |
| 13-24 months | 17,079 | 5,251 | 913 | 17.4% | 44.6% | 36.9% |
| over 24 months | 10,158 | 2,927 | 580 | 19.8% | 24.9% | 23.4% |

The strict top-20% policy flags **8,167 customers**, rounding the customer budget down. Its recall is **0.339**. Equal scores are ordered by CustomerID.

Model choice and threshold choice both use these development outcomes, so F1 and missed-churner counts may be optimistic. The size of that effect has not been measured. Campaign decisions also require a contact budget or cost objective. The final independent test evaluation requires fixed modelling and threshold choices.

**Validation and run record**

- All notebook code cells completed successfully.
- No dependency changes were needed. The committed data and train/test split were left unchanged.
- Python 3.12.16; scikit-learn 1.9.1; NumPy 2.5.4; pandas 3.1.0. Four CPU cores, OpenMP four threads, BLAS two threads.
- Gradient Boosting used 155 rounds on the first fold, with a separate internal 10% stopping sample. Stopping monitors log loss; outer scoring uses AP.
- Split SHA-256: `ba3fd8e7893a415c5f7b608e61bde014004bf17d91fd6bd47da948f8c4d3b729`.

Source hashes for this recorded run:

- `notebooks/modelling.py`: `b2ade2187a845df50d6be4decfab595f30b1178dfd3c3446a0df16362ec4f7c7`.
- `src/__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `src/baselines.py`: `1b4bdb5621fc6e17363d9b0c5af9a0c7497bc3e5e56260e4da63d0e1d388cfb9`.
- `src/data.py`: `184d1fb12457c35ec82115bc16f798163a0c9fe6e463579cd806c7f2545bf19e`.
- `src/eda_utils.py`: `0b65216d4603c33c31480d7f763080e87a443e5a5dd8508418400815e5db9f6c`.
- `src/evaluation.py`: `12244d793428e4393ec4940e430bbe44236e893ab1d4d6b31cc4e314c455729f`.
- `src/features.py`: `0db9e7d5dd068d60058d162d77ea1832215b3d27f72ceb835a0b865314646286`.
- `src/group_comparison.py`: `a72eab73cf7da32f7184cfade843aec44ea73217b0d1d0946e99e22f51ca7aa6`.
- `src/models.py`: `f20e62b6b3f28d9b9a853fd3e74c03c21669c63b114555f58777e28ffd11bfd4`.
- `src/preprocessing.py`: `8c88ce5bd011c9a24957139378d9b11f5274b1b020af86907dd7d0c37a69fa85`.
- `src/thresholds.py`: `c89a786b0984c108894aa425fdf8329f86d20156646cca0cf984a1474b81982d`.
