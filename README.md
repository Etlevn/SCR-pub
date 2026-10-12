# SCR: Personalized Stock Candidate Ranking

Personalized Stock Candidate Ranking with Machine Learning and Collaborative Filtering

This project treats an investor's stock choice as **candidate-set ranking**. For each investor on a given trading day, the model scores and ranks the candidate stocks, keeps about the top 10%, and checks whether the stock that was actually traded remains in that set. The main metric is **Recall@10%**.

The study proceeds through three questions:

1. **Feature-based machine learning:** Can 14 stock features support effective ranking?
2. **Collaborative filtering:** Can historical user–stock interactions provide a ranking signal that generalizes on its own?
3. **Hybrid models:** When the two signals are combined, can they beat the best feature-only model?

Under one time split and one Recall@10% definition, the comparison covers **13 machine-learning models, 7 collaborative-filtering models, and 36 hybrid models: 56 in total**.

## Problem definition

Each evaluation group is an investor–date candidate set, keyed by `(accid, date, now_permno)`.

- `permno`: one candidate stock in the group
- `now_permno`: the stock the investor actually traded on that day
- Positive label: `permno == now_permno`

Model outputs are used only as within-group ranking scores. There is no fixed classification threshold. The pipeline is:

**Candidate set → score → rank within the group → keep about the top 10% → was the traded stock kept?**

When a group has a single positive example, group-averaged Recall@10% is the hit rate at that cutoff.

The number kept is `k = max(1, int(group size × 0.1))`. Very small candidate sets always keep at least one stock, so the retained share is about 10%.

## Data and split

The data file is `data/master_final_sampled.dta`. The pipeline keeps the identifier columns and `f0`–`f13`, then builds `label` from them. This repository does not impute missing values or refit a scaler. The 14 features are used as stored in the file.

The split is chronological inside each `accid`: after sorting dates, the first 70% of dates go to training and the rest go to test. 70/30 is the target ratio within each user. The pooled group counts do not have to be exactly 70/30.

Evaluation-group counts aligned with the saved results:

| | All groups | Groups with a positive |
|--|-----------:|-----------------------:|
| Train | 453 | 217 |
| Test | 252 | 150 |
| Total | 705 | 367 |

Of the 150 test groups that contain a positive, 128 belong to users seen in training (warm-start) and 22 belong to users not seen in training (cold-start). Those counts are the Recall denominators. The full set of 252 test groups also includes groups with no positive. Those groups enter the Success Rate denominator and stay out of the Recall denominator.

Collaborative filtering writes only positive training interactions into the user–stock matrix. That matrix is more than 99.9% sparse. Users or stocks unseen at test time fall back to popularity. Popularity is not an eighth model in the comparison.

## Metrics

Within each group, candidates are sorted by score from high to low. A hit means the positive example is among the top `k`.

- **Recall@10%** = hits / groups that contain a positive
- **Success Rate** = hits / all evaluation groups

Success Rate is therefore lower than Recall. The best machine-learning models reach 139/150 = 92.67% Recall on the test set, and 139/252 = 55.16% Success Rate.

Warm-start and cold-start Recall are computed only in the collaborative-filtering and hybrid experiments. A cold-start user is an `accid` absent from the training interaction matrix. In the machine-learning results those two columns are fixed at 0, because that pipeline does not split the metric.

92.67% and 86.36% have different denominators. The first is all 150 test groups that contain a positive. The second is the 22 cold-start groups among them. The two rates are separate subset hit rates.

## Stage 1: Feature-based machine learning

All 13 models use scikit-learn's default hyperparameters, with `random_state=42`. Ranking scores use the positive-class probability when it exists, otherwise `decision_function`. Regression models use the predicted value.

| Model | Train Recall | Test Recall | Test Success |
|------|-------------:|------------:|-------------:|
| logistic_l2 | 88.94% | **92.67%** | 55.16% |
| linear_svm | 90.78% | **92.67%** | 55.16% |
| ridge_classifier | 90.32% | **92.67%** | 55.16% |
| ridge | 90.32% | **92.67%** | 55.16% |
| sgd_log | 88.94% | 92.00% | 54.76% |
| logistic_l1 | 88.02% | 91.33% | 54.37% |
| logistic_enet | 88.02% | 91.33% | 54.37% |
| sgd_modified_huber | 85.71% | 91.33% | 54.37% |
| passive_aggressive | 74.19% | 79.33% | 47.22% |
| perceptron | 71.43% | 78.00% | 46.43% |
| sgd_hinge | 68.66% | 73.33% | 43.65% |
| lasso | 14.29% | 13.33% | 7.94% |
| elasticnet | 14.29% | 13.33% | 7.94% |

The best results come from L2-regularized linear models: keeping about 10% of the candidates still covers 139/150 traded stocks. Strongly L1-regularized Lasso and ElasticNet fall to 13.33%. This set of results is the feature baseline for the later methods.

## Stage 2: Collaborative filtering

The seven models use only positive interactions from the training period, then map scores back onto the same candidate sets and Recall@10%.

| Model | Train Recall | Test Recall | Warm-start | Cold-start |
|------|-------------:|------------:|-----------:|-----------:|
| svd | 100% | **43.33%** | 43.75% | 40.91% |
| nmf | 100% | **43.33%** | 43.75% | 40.91% |
| als | 100% | 42.67% | 42.97% | 40.91% |
| bpr | 100% | 41.33% | 41.41% | 40.91% |
| item_cf | 100% | 41.33% | 41.41% | 40.91% |
| knn | 100% | 41.33% | 41.41% | 40.91% |
| user_cf | 29.95% | 24.67% | 21.88% | 40.91% |

On the test set, SVD / NMF hit 65/150, ALS hits 64/150, BPR, Item-CF, and KNN hit 62/150, and User-CF hits 37/150. Matrix factorization hits every training group that contains a positive, while test recall stays near 43%, well below the 92.67% feature baseline.

Every model's cold-start recall is 9/22 = 40.91%, matching the popularity fallback. User-CF warm-start recall (28/128 = 21.88%) is below that fallback.

## Stage 3: Hybrid models

The top three test models from each side are crossed:

- Machine learning: `linear_svm`, `logistic_l2`, `ridge_classifier`
- Collaborative filtering: `nmf`, `als`, `svd`

Four fusion strategies, nine combinations each, give 36 models. Names follow `{strategy}_{cf}_{ml}`.

| Strategy | What it does | Test Recall |
|------|------|------------:|
| weighted | Min–max scale each score stream to [0, 1] inside the group, then add 0.3×CF + 0.7×ML. The weight is fixed; there is no weight search | **92.67%** for all 9 |
| stacking | The named CF model and ML model are each one base model. The meta-learner is always `logistic_l2`. Meta-features are those base models' predictions on the same training set, not time-held-out predictions | **92.67%** for all 9 |
| feature_cf | Truncate user and stock latent vectors to at most 50 dimensions each, append them to the original 14 features, and train the ML model. Embeddings for cold-start users or stocks are zeros | 92.67% for 5/9; the rest are 92.00% or 91.33% |
| switching | If the user has at least 3 training interactions, score the whole group with CF; otherwise score it with ML | **60.00%–62.67%** |

Weighted and stacking hybrids hit the same 139/150 test cases as the best machine-learning models. They do not raise out-of-sample recall. Under switching, users with enough interactions are sent to the weaker collaborative-filtering model, and test recall falls to about 60%–63% (the best combination is 94/150).

On the 22 cold-start groups that contain a positive, hybrid recall is 19/22 = 86.36%. On the 128 warm-start groups that contain a positive, weighted and stacking hybrids reach 120/128 = 93.75%. These are hit rates on two different subsets.

## Conclusions

- With the current features, L2 linear models keep about 10% of the candidate pool and still retain 92.67% of the traded stocks.
- The user–stock matrix is more than 99.9% sparse. Collaborative filtering memorizes training interactions and reaches about 43% test recall. User-CF reaches 24.67%.
- The tested weighted and stacking hybrids do not beat the best feature model. Switching is lower. A more complicated fusion did not produce a higher Recall@10%.
- This comparison covers the 56 implemented configurations. It does not establish a theoretical ceiling for every possible fusion method.

## Layout

```
SCR/
├── ml_main.py  cf_main.py  hy_main.py
├── README.md
├── requirements.txt
├── data/          # master_final_sampled.dta and the loader
├── config/        # ml / cf / hy configuration
├── prep/          # loading, labels, time split, interaction matrix
├── model/         # machine learning, collaborative filtering, hybrids
└── out/           # result CSVs and the result writer
```

Each pipeline is: configuration → load and split → train → group-wise Recall@10% → write CSV. The shared split ratio, random seed, and data path live in `config/ml_config.py`. Run the entry scripts from the project root.

## Run

```bash
pip install -r requirements.txt

python ml_main.py
python cf_main.py
python hy_main.py
```

Dependencies are Python 3.8+, pandas, scikit-learn, scipy, implicit, numpy, and tqdm. Run from the project root. Place the data file at `data/master_final_sampled.dta`. Results are written to `out/ml_results.csv`, `out/cf_results.csv`, and `out/hy_results.csv`.

## Reproduction notes

The tables above come from the result files saved in this repository. Group counts are recovered from the hit rates and match integer hit counts, for example 139/150 for the best test recall and 193/217 for training `logistic_l2`. Rerunning overwrites those CSVs. Different data or dependency versions can change the numbers.
