# COVID-19 Misinformation Detection

**One proposed pipeline: CT-BERT with URL and emoji preprocessing and sigmoid calibration.** It achieved **98.457% weighted F1 on validation** and **98.318% on the CONSTRAINT 2021 test mirror**.

[Live Streamlit demo](https://broad-cast-lexington-penalty.trycloudflare.com) · [Full metrics](reports/20261124_ctbert_coursework/metrics_all_validation_test.csv) · [Vietnamese analysis](reports/20261124_ctbert_coursework/phan_tich_ket_qua.md) · [Streamlit guide](apps/coursework_v1/README.md)

## Why we chose this pipeline

COVID-19 posts are short, noisy, and often include links or emoji. We started with TF-IDF classifiers, graph models, and general-purpose transformers, then asked whether a model pretrained on COVID-19 tweets could make better use of this format.

CT-BERT improved on our general-purpose transformer baselines. We then tested whether combining CT-BERT with DistilBERT helped. Weighted averaging and Choquet improved some probability scores, but neither reduced the total number of test errors versus CT-BERT alone. That led us back to the text itself: replace URLs with a shared token, turn emoji into text, fine-tune CT-BERT, and calibrate its output with a sigmoid.

We selected this single pipeline by validation weighted F1. It reduced errors from **38 to 33** on validation and from **37 to 36** on the test mirror versus calibrated CT-BERT. The paired bootstrap interval for the test F1 change spans zero, so the small observed gain needs confirmation on independent data. This is a project pipeline assembled from established methods.

## Proposed method vs. our baselines

Each member model used the same 3,792-example fit split. Neural checkpoint selection used 702 separate inner-validation examples. Calibration and learned pooling used separate 642-example sets. The final tables use 2,139 validation examples and 2,140 examples in the test mirror. The comparison below shows the configurations available in the app. The full CSV also includes the remaining experimental controls.

| Role | Configuration | Validation weighted F1 | Test mirror weighted F1 |
|---|---|---:|---:|
| **Proposed** | **CT-BERT with URL and emoji preprocessing** | 98.457% | 98.318% |
| Ensemble control | CT-BERT + DistilBERT · Trung bình có trọng số | 98.270% | 98.271% |
| Baseline | CT-BERT | 98.223% | 98.271% |
| Ensemble control | CT-BERT + DistilBERT · Choquet | 98.176% | 98.271% |
| Ensemble control | SVM + DistilBERT · Trọng số | 96.398% | 95.466% |
| Baseline | DistilBERT | 96.304% | 95.138% |
| Ensemble control | SVM + DistilBERT · Choquet | 96.258% | 95.933% |
| Baseline | BERT-Base | 95.885% | 95.888% |
| Baseline | RoBERTa-Base | 95.420% | 94.909% |
| Ensemble control | SVM + Logistic ký tự · Choquet | 94.104% | 94.528% |
| Ensemble control | SVM + Logistic ký tự · Trọng số | 94.057% | 94.574% |
| Baseline | Word TF-IDF + LinearSVC | 93.400% | 93.734% |
| Baseline | Character TF-IDF + Logistic Regression | 93.400% | 94.062% |
| Baseline | GPT-2 classifier | 92.705% | 92.290% |
| Baseline | Word TF-IDF + Logistic Regression | 91.997% | 92.330% |
| Baseline | BERT-Tiny | 91.896% | 91.716% |
| Baseline | Word TF-IDF + Multinomial NB | 91.537% | 91.072% |
| Baseline | GCN | 87.689% | 88.968% |
| Baseline | GAT | 87.221% | 88.546% |

## Comparison with prior work

| Study | Configuration | Test source | Weighted F1 |
|---|---|---|---:|
| [Felber (2021)](https://arxiv.org/abs/2101.03717) | Linear SVM with linguistic features | Official shared-task test | 95.19% |
| **This project** | **CT-BERT with URL and emoji preprocessing** | **CONSTRAINT test mirror** | **98.318%** |
| [Glazkova et al. (2021)](https://arxiv.org/abs/2012.11967) | Majority vote across three CT-BERT models | Official shared-task test | 98.69% |

These are benchmark references across different training protocols. The mirror matches the documented sample count and class distribution. Row-level equivalence with the official release remains unverified. The project used a single seed and had previously evaluated the mirror. [Dataset description](https://arxiv.org/abs/2011.03327)

![CT-BERT configurations across the two evaluation splits](figures/presentation/08_validation_va_test_ctbert.png)

## Try the Streamlit app

[Open the live Streamlit demo](https://broad-cast-lexington-penalty.trycloudflare.com). This temporary link works while the host PC, Streamlit, and its tunnel are running.

The app defaults to the proposed pipeline for text prediction, CSV batches, and LIME. Baselines remain selectable for comparison. This is a text-classification demo rather than a medical fact-checking service.

Run locally from the repository root with Python 3.11:

```bash
python -m pip install -r apps/coursework_v1/requirements.txt
python -m streamlit run apps/coursework_v1/app.py
```

The dashboard reads the aggregate results from this repository. Local inference requires the pinned checkpoint, tokenizer, calibrator, and lock files listed in the [model registry](apps/coursework_v1/model_registry.json). Transformer inference uses CUDA. Checkpoints and raw benchmark data are kept outside this public repository.

## Files for the team

- [Notebooks 18–22](notebooks/) record the evaluation and CT-BERT experiments.
- [Presentation figures](figures/presentation/) include learning curves, confusion matrices, and reliability diagrams.
- [Team handoff](reports/20261124_ctbert_coursework/HUONG_DAN_CHIA_SE.md) points to the material for the written report.
