# CSCI167 Deep Learning Project: Rotten Tomatoes sentiment classification

**Dataset:** `cornell-movie-review-data/rotten_tomatoes` (Pang & Lee, ACL 2005). Binary sentiment (0 = neg, 1 = pos) on single sentences.
Official splits: train 8,530 / validation 1,066 / test 1,066. Balanced (5,331 pos / 5,331 neg overall). License listed as "unknown" on the dataset card.

**Research question (proposed, confirm as a group):** How much does pretraining help compared with training from scratch on a small text dataset, and how does the gap change as training data shrinks (100% / 25% / 10%)?

## Layout
| File | Purpose |
|---|---|
| `common.py` | Shared, FROZEN code: data/splits, vocab, training loop, early stopping, logging, plots |
| `notebooks/A_baseline_mlp.ipynb` | Person A template: write the baseline (embedding + fully connected network) |
| `notebooks/B_cnn_hparams.ipynb` | Person B template: write a 1D CNN, then run the hyperparameter study (>= 2 factors, >= 2 settings each) |
| `notebooks/C_transformers.ipynb` | Person C template: write a small transformer (scratch); DistilBERT fine-tuned/frozen is wired up |
| `results/` | `results_<A|B|C>.csv`, `history/*.json` (learning curves + test predictions), `checkpoints/`, `env_*.json` |

Each notebook is a **template**: the setup, training runs, sanity check, plots and error analysis already work.
You write the model class (`MyModel`, cell marked `YOUR MODEL`, with suggested architecture and textbook chapter in the text above it).

## Run (Google Colab)
1. Repo: https://github.com/ZhongVang/csci167-rotten-tomatoes (`REPO_URL` in each notebook's first code cell already points to it).
2. Open the notebook in Colab, set Runtime to GPU (T4), Run all. Results are saved to Drive (`MyDrive/CSCI167_results`) if you allow the Drive prompt.
3. After everyone has run: copy the three `results_*.csv` into one folder and run `summarize()` (see end of each notebook).

Local: `pip install -r requirements.txt`, then open the notebooks.

## Fixed protocol (all models, so the comparison is fair)
- Splits: official; train subsets are stratified random subsets of TRAIN only (seeded). Val/test never subsampled.
- Vocab built from the (sub)sampled TRAIN text only (min freq 2, max 20k, max length 64 tokens).
- Seeds 0, 1, 2 for every configuration; report mean +/- std.
- From-scratch models: Adam lr 1e-3, batch 64, max 20 epochs, early stopping patience 5 on **validation accuracy**; best-validation weights restored.
- DistilBERT: lr 2e-5, batch 32, max 4 epochs, patience 2 (differences in pretraining/budget must be stated in the report).
- Test set is evaluated exactly once per run, after model selection. Metrics: accuracy (primary), macro-F1.
- Do not edit `common.py` after runs start; each CSV row stores `common_version` (hash of the file).

## To do for the report (rubric checklist)
- [ ] Dataset section: task, sizes, input format, labels, balance, license, revision (`results/env_*.json`).
- [ ] Models: structure, trainable parameter counts (`trainable_params`), rationale; what is frozen/trained for DistilBERT.
- [ ] Hyperparameter study (B): optimizer {Adam, SGD} and dropout {0.0, 0.5}; effects on convergence, val accuracy, stability, time.
- [ ] Learning curves (`plot_history`), results table (`summarize`), error analysis (end of each notebook).
- [ ] Reduced-data results (10% / 25% / 100%); robustness claims need a separate experiment (e.g. noisy text).
- [ ] Limitations: small noisy test set (~+/-1.5%), only 3 seeds, no official split beyond the random one, different budgets for pretrained model.
- [ ] Working links to notebooks/repo; check instructor access (Anyone with link: Viewer).
