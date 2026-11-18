# CSCI167 Deep Learning Project: Rotten Tomatoes sentiment classification

**Dataset:** `cornell-movie-review-data/rotten_tomatoes` (Pang & Lee, ACL 2005). Binary sentiment (0 = neg, 1 = pos) on single sentences.
Official splits: train 8,530 / validation 1,066 / test 1,066. Balanced (5,331 pos / 5,331 neg overall). License listed as "unknown" on the dataset card.

**Research question (proposed, confirm as a group):** How much does pretraining help compared with training from scratch on a small text dataset, and how does the gap change as training data shrinks (100% / 25% / 10%)?

## Layout
| File | Purpose |
|---|---|
| `common.py` | Shared, FROZEN code: data/splits, vocab, training loop, early stopping, logging, plots |
| `notebooks/model_template.ipynb` | Shared template: setup, data, training settings, run function, results table, learning curves, error analysis. Everyone copies it and adds their own architecture |
| `results/` | `results_<yourname>.csv`, `history/*.json` (learning curves + test predictions), `checkpoints/`, `env_*.json` |

The architectures are not chosen yet. In your copy of the template, define your model in the section marked **YOUR ARCHITECTURE**
(a class `MyModel(nn.Module)` with `forward(ids, mask)` returning `[batch, 2]` logits; details are in the notebook). Everything else already works.

## Run (Google Colab)
1. Repo: https://github.com/ZhongVang/csci167-rotten-tomatoes (`REPO_URL` in the template's first code cell already points to it).
2. Copy `model_template.ipynb`, set `OWNER`, `TAG` and `MODEL_NAME`, add your model, set Runtime to GPU (T4), Run all. Results are saved to Drive (`MyDrive/CSCI167_results`) if you allow the Drive prompt.
3. After everyone has run: copy all `results_*.csv` into one folder and run `summarize()` (see the end of the template).

Local: `pip install -r requirements.txt`, then open the notebooks.

## Fixed protocol (all models, so the comparison is fair)
- Splits: official; train subsets are stratified random subsets of TRAIN only (seeded). Val/test never subsampled.
- Vocab built from the (sub)sampled TRAIN text only (min freq 2, max 20k, max length 64 tokens).
- Seeds 0, 1, 2 for every configuration; report mean +/- std.
- Shared training settings: Adam lr 1e-3, batch 64, max 20 epochs, early stopping patience 5 on **validation accuracy**; best-validation weights restored.
- Any model that needs a different budget (e.g. a pretrained model) must state the difference in the report.
- Test set is evaluated exactly once per run, after model selection. Metrics: accuracy (primary), macro-F1.
- Do not edit `common.py` after runs start; each CSV row stores `common_version` (hash of the file).

## To do for the report (rubric checklist)
- [ ] Dataset section: task, sizes, input format, labels, balance, license, revision (`results/env_*.json`).
- [ ] Models: structure, trainable parameter counts (`trainable_params`), rationale; what is frozen/trained for any pretrained model.
- [ ] Hyperparameter study (group decides who runs it): at least 2 factors, at least 2 settings each; effects on convergence, val accuracy, stability, time.
- [ ] Learning curves (`plot_history`), results table (`summarize`), error analysis (end of each notebook).
- [ ] Reduced-data results (10% / 25% / 100%); robustness claims need a separate experiment (e.g. noisy text).
- [ ] Limitations: small noisy test set (~+/-1.5%), only 3 seeds, no official split beyond the random one, different budgets for pretrained model.
- [ ] Working links to notebooks/repo; check instructor access (Anyone with link: Viewer).
