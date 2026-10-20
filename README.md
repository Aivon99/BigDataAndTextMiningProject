# Chessboard Understanding with a Vision-Language Model

Project for the **Big Data Analytics and Text Mining** course at the **University of Bologna**.

- Professor: Gianluca Moro, University of Bologna
- Tutor: Ing. Giacomo Frisoni

We fine-tune a small vision-language model (Qwen3.5-0.8B) with LoRA to read chessboard images,
and we study how the order in which the model reads the image tokens changes the results,
following the idea of the [REOrder](https://github.com/d3tk/REOrder) paper (NeurIPS 2025).

## Hugging Face

Datasets, trained models and results are stored in the
[bdatm-project](https://huggingface.co/bdatm-project) organization on Hugging Face.

| What | Repositories |
|---|---|
| Datasets | [dataset_task1](https://huggingface.co/datasets/bdatm-project/dataset_task1), [dataset_task2](https://huggingface.co/datasets/bdatm-project/dataset_task2), [dataset_task3](https://huggingface.co/datasets/bdatm-project/dataset_task3) |
| LoRA models | `qwen-task{1,2,3}-{strategy}-tok256-lora`, for example [qwen-task1-raster-tok256-lora](https://huggingface.co/bdatm-project/qwen-task1-raster-tok256-lora) |
| Results | [evaluation-results-task1](https://huggingface.co/datasets/bdatm-project/evaluation-results-task1), [evaluation-results-task2](https://huggingface.co/datasets/bdatm-project/evaluation-results-task2), [evaluation-results-task3](https://huggingface.co/datasets/bdatm-project/evaluation-results-task3) |

The strategy in the model name is `raster`, `zigzag`, `spiral_center_out`, `file_wise`,
`hilbert` or `learned`.

The repositories without `-tok256` (`qwen-task{1,2,3}-{standard,zigzag,spiral,file_wise,learned-reordering}-lora`)
come from our first implementation, which reordered pixel tiles of 512px images (see
[Legacy runs](#legacy-runs)). Their results are kept in the results tables, labeled
`legacy: pixel-tile reordering, 512px`.

## Tasks

| Task | Input | Output |
|---|---|---|
| Task 1 | One board image | The position in FEN |
| Task 2 | One board image with the last move highlighted | That move in SAN |
| Task 3 | Two board images, before and after a move | That move in SAN |

## Dataset

The data comes from the [Lichess puzzle database](https://database.lichess.org/#puzzles).
From each puzzle we take the starting position and the first move.

- 20,000 puzzles are sampled from the CSV, then 2,000 with white to move and 2,000 with black
  to move are kept (4,000 in total).
- The puzzles are split 80/10/10 into train, validation and test (3,200 / 400 / 400), stratified
  by side to move.
- There is one dataset for each task, and all three are built from the same 4,000 puzzles.
  Only the images and the expected answer change from one task to another.
- Boards are drawn with python-chess and cairosvg as 512x512 images without coordinates, so the
  8x8 squares fill the whole image. Task 2 uses python-chess's last-move highlight.

These numbers can be changed in `configs/config.yaml`.

Lichess replaces its puzzle database regularly, so downloading it from Lichess today gives a newer
file and a different sample. The version we used is kept as revision `5dc959d3f4` of
[Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles), Lichess's mirror on
Hugging Face: with it, `Datasets_creator.ipynb` selects exactly our puzzles, with the same splits and
order (a comment in its download cell shows how). We checked the puzzles and the splits, not the
images: they are drawn again, and other versions of python-chess or cairosvg could change a few
pixels. The exact datasets we used are on Hugging Face (see the links above), and all the
experiments use them.

## Method

**Image resolution.** The model's processor resizes every board to 256x256. The vision encoder
cuts it into 16px patches (a 16x16 grid) and merges them 2x2, so the language model receives an
8x8 grid of visual tokens: exactly one token per chess square. 512px (2x2 tokens per square) is
used as a resolution ablation.

**Token reordering.** A scan order is applied to the 64 visual tokens inside the model: the token
embeddings and their M-RoPE positions are permuted together. The image is never changed and every
token keeps its true 2D position; only the order in which the language model reads the tokens
changes. This matters for Qwen3.5 because 18 of its 24 language-model layers are Gated DeltaNet,
a recurrent scan with no positional encoding.

The scan orders are:

- `raster`: left to right, top to bottom (the default order, also called rank-wise)
- `zigzag`: like raster, but every other rank is read right to left
- `spiral_center_out`: a spiral starting from the center of the board
- `file_wise`: column by column
- `hilbert`: a Hilbert curve, taken from the REOrder code

For each task we compare:

- **Vanilla**: the base model with no training, zero-shot and 3-shot (3 solved training examples
  in the prompt).
- **LoRA**: the model fine-tuned with LoRA in raster order.
- **Training-free reordering (TF)**: the raster LoRA model reads the same images in one of the other
  orders, with no new training.
- **Fine-tuned reordering (SFT)**: a new LoRA model is trained and tested with each order.
- **Learned reordering**: REOrder's Plackett-Luce policy learns an order over the 64 squares,
  trained together with LoRA using REINFORCE. The first epoch is a warm-up in raster order; then
  the policy samples orders. Training runs for up to 10 epochs with early stopping on the
  validation loss (patience 2), and the best epoch is kept: in all three tasks training stopped
  after 6 epochs and the best epoch was the 4th. The REOrder code is
  downloaded automatically into `external/REOrder` (at a fixed commit) the first time it is needed.
- **Learned order, frozen policy**: REOrder's last phase. The order learned by the policy is fixed,
  and the LoRA model of the learned-reordering run keeps training with that order.

LoRA settings: r = 16, alpha = 32, dropout 0.05 on all attention and MLP projections,
learning rate 2e-4, batch size 8, up to 10 epochs with early stopping on the validation loss
(the best epoch is kept).

Metrics:

- Task 1: FEN exact match, character error rate, Levenshtein distance and square-by-square
  accuracy, plus separate checks for each part of the FEN. The main score is the exact match on
  the board part of the FEN: side to move, castling rights and move counters cannot be seen in
  the image, so the full-string exact match is at most 3% for every model.
- Task 2 and Task 3: exact match on the SAN move, taken from the first line of the model's answer,
  plus extra checks such as whether the predicted move is legal or is the same move written in a
  different way.

## Results

All numbers are percentages on the 400 test samples, at 256px with token reordering. Task 1 is
the board exact match; Tasks 2 and 3 are the SAN exact match, with "same move" (the right move,
written in any notation) in brackets.

| Model | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Vanilla, zero-shot | 0.00 | 0.00 (0.00) | 0.25 (0.25) |
| Vanilla, 3-shot | 0.00 | 0.75 (0.75) | 0.00 (0.00) |
| LoRA, raster | 97.00 | **60.25** (93.75) | 79.75 (88.50) |
| LoRA, zigzag | 97.25 | 53.50 (89.50) | 77.50 (86.00) |
| LoRA, spiral_center_out | **98.50** | 54.00 (82.50) | 70.75 (82.25) |
| LoRA, file_wise | 96.75 | 54.75 (88.50) | **80.50** (93.25) |
| LoRA, hilbert | 95.50 | 52.50 (85.75) | 67.75 (83.75) |
| LoRA, learned order | 95.75 | 57.75 (89.25) | 57.25 (81.00) |
| LoRA, learned order, frozen policy | 96.75 | 59.50 (88.25) | 75.00 (86.00) |

Training-free reordering (the raster LoRA model read in another order):

| Order | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| zigzag | 0.00 | 52.75 | 42.25 |
| spiral_center_out | 0.00 | 14.50 | 5.75 |
| file_wise | 0.00 | 9.75 | 4.00 |
| hilbert | 0.00 | 29.00 | 12.75 |

Resolution ablation (raster LoRA, same training recipe):

| Resolution | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| 256px (1 token per square) | 97.00 | 60.25 | 79.75 |
| 512px (2x2 tokens per square) | 98.75 | 62.00 | 87.50 |

Main findings:

- LoRA turns the model into a reliable board reader: from 0% to 97% on Task 1. Square-by-square
  accuracy is above 97.9% for every model trained with its own order.
- The order matters at test time: a model trained in raster order collapses when it reads the
  tokens in any other order (0% on Task 1).
- When a model is trained with an order, no fixed order wins on every task (spiral_center_out on
  Task 1, raster on Task 2, file_wise on Task 3).
- The learned order stays close to raster and does not beat it.
- 512px is better than the 1:1 grid, especially on Task 3: more tokens per square help more than
  the exact alignment.
- On Task 2, most errors of the raster model are notation only. In 119 of its 121 "same move,
  different notation" errors, the only difference is `x`, `+` or `#`. A single image taken after
  the move cannot show whether a piece was captured, and 147 of the 400 test moves are captures.
- On Task 3 (raster model, 100 test samples), swapping the two frames or giving the same frame
  twice raises the answer loss from 0.16 to 1.4-2.2 (about 9-13 times), and the correct order has
  the lowest loss on 94% of the samples. So the model really compares the two images.

The full tables (with all metrics) are on Hugging Face:
[Task 1](https://huggingface.co/datasets/bdatm-project/evaluation-results-task1),
[Task 2](https://huggingface.co/datasets/bdatm-project/evaluation-results-task2),
[Task 3](https://huggingface.co/datasets/bdatm-project/evaluation-results-task3).

### Legacy runs

Our first implementation rearranged 64px tiles of the 512px image before the vision encoder, so
a reordered image was a different picture of the board (for example, `file_wise` gave a
transposed board). These runs are kept in the results tables and in the repositories without
`-tok256`. Their raster run is the 512px row of the resolution ablation; with raster order the
image is not changed, so that comparison is fair. `Reproduce_results.ipynb` evaluates this run
again (see [How to verify the results](#how-to-verify-the-results)).

## Repository structure

```
configs/
  config.yaml              settings for the dataset generation
notebooks/
  Datasets_creator.ipynb   builds the three datasets and uploads them to Hugging Face
  Task_1.ipynb             training and evaluation for Task 1
  Task_2.ipynb             training and evaluation for Task 2
  Task_3.ipynb             training and evaluation for Task 3
  Reproduce_results.ipynb  evaluates the published models again and compares them with our results
src/
  data/
    generation.py          draws the boards and builds the samples
    utilities.py           loads the config and the Lichess CSV
  eval/
    metrics.py             FEN and SAN metrics
    evaluation.py          runs a model on the test set of each task
    results.py             saves and loads the results on Hugging Face
    diagnostics.py         extra analysis of the errors and of the training runs
  training/
    checkpoints.py         saves and resumes training runs on Hugging Face
    preprocessing.py       turns dataset rows into model inputs
    resolution.py          sets the image resolution of the processor (256px = 1 token per square)
    reordering.py          the scan orders
    token_reordering.py    applies a scan order to the visual tokens inside the model
    reorder_lib.py         downloads and loads the REOrder code
    finetune.py            LoRA fine-tuning for one task and one scan order
    learned_reordering.py  learned reordering with REOrder's Plackett-Luce policy
requirements.txt
```

## How to verify the results

`notebooks/Reproduce_results.ipynb` checks that the code produces the numbers in the tables above,
without retraining. It downloads the LoRA adapters we published on Hugging Face, evaluates them
again on the test set from scratch (no cached results are used) and compares the new predictions
with the ones we saved, sample by sample.

1. Open the notebook in Google Colab (*File > Open notebook > GitHub*, or upload it) and choose an
   **L4** or **A100** GPU (*Runtime > Change runtime type*). The model runs in bfloat16, which the
   free T4 GPU does not support well.
2. Choose `LEVEL` in Section 2 of the notebook (on Colab it is a drop-down menu). The outputs
   saved in the notebook come from `"tables"`; for a quick check choose `"main"`:

   | `LEVEL` | What runs | Time on an L4 |
   |---|---|---|
   | `"main"` (quick check) | The raster LoRA model of each task, on all 400 test samples | about 35 min |
   | `"tables"` (the saved outputs) | Every row of the tables above, including the 512px resolution ablation. Task 2 and Task 3 on all 400 samples; Task 1 on all 400 samples for the raster LoRA model and on the first 100 for the other rows | about 2.5 h |
   | `"full"` | Every row of the tables above, including the 512px resolution ablation, on all 400 test samples | about 6.5 h |

3. Run all the cells (*Runtime > Run all*).

The times above are for the evaluation. Without a token, downloading the three datasets from
Hugging Face takes some extra time: about 10 minutes each for Task 1 and Task 2 and about 20
minutes for Task 3 (two images per sample), measured on our connection.

No Hugging Face token is needed: all our datasets, models and result tables are public, and the
notebook does not write anything to the Hub. The last cell shows, for each model, our score, the
reproduced score on the same samples and how many predictions are identical. Decoding is greedy, so
on the same GPU type the predictions should be identical or differ on one or two samples; on a
different GPU type a few more can change.

To run it locally instead, install the packages with `pip install -r requirements.txt`, set
`CONFIG["colab"] = False` in the notebook and start Jupyter from the `notebooks/` folder.

**The 512px resolution ablation.** The task notebooks run at 256px only. The 512px raster LoRA
models were trained with the same training recipe as the 256px ones (same hyperparameters,
preprocessing, LoRA settings and early stopping); the only difference is the image size. They were
trained with an earlier version of our code, which reordered pixel tiles of the image (see
[Legacy runs](#legacy-runs)). In raster order that version leaves the image unchanged, so for these
models it does the same thing as the current code at 512px. Their adapters are
`qwen-task{1,2,3}-standard-lora`, and `Reproduce_results.ipynb` evaluates them again at 512px with
the current code (`LEVEL = "tables"` or `"full"`), so the 512px numbers can be checked like all the
others. To train a 512px model with the current code, set `IMAGE_SIZE = 512` in a task notebook.

## Tested environment

Our runs used Google Colab with an NVIDIA L4 GPU. The task notebooks install the newest release
of each package, so the versions below are the releases that were current during our final runs
(October 2026). They are pinned in `requirements.txt`, and `Reproduce_results.ipynb` installs
exactly these versions.

| Package | Version |
|---|---|
| transformers | 5.20.2 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 5.1.0 |
| huggingface_hub | 2.2.0 |
| flash-linear-attention | 0.5.2 |
| torch | 2.14.1 for Task 1 (the notebook upgrades it); Colab's preinstalled version for Tasks 2 and 3 |

Qwen3.5 needs transformers 5.2.0 or newer. The token reordering patches internal methods of the
model in transformers, so other versions may break it; `verify_token_reordering` checks that it
works, in every task notebook and in `Reproduce_results.ipynb`.

## Full retraining

**We do not recommend retraining to check our results**: `Reproduce_results.ipynb` already
verifies them with our datasets and our trained models (see
[How to verify the results](#how-to-verify-the-results)). Retraining everything means 21 training
runs (7 per task), each of 6 to 9 epochs on 3,200 samples, which takes many hours of GPU time.

The task notebooks contain the whole pipeline: training, evaluation and analysis. They are the
record of our runs, with their outputs. They are meant to run on Google Colab with a GPU, but they
also work locally. To retrain:

1. **Use your own Hugging Face namespace.** Every trained model and every result table is saved on
   the Hub, under the namespace set by `hf_org_prefix` in Section 3 of each task notebook. Ours is
   `bdatm-project`, where you cannot write; change `hf_org_prefix` to your Hugging Face username
   or organization. If you keep `bdatm-project`, the notebooks find our finished runs and load them
   instead of training.
2. **Add a Hugging Face token with write access** to that namespace: on Colab as a secret called
   `HF_TOKEN`, locally in a `.env` file in the project folder (`HF_TOKEN=hf_...`).
3. Run `Task_1.ipynb`, `Task_2.ipynb` and `Task_3.ipynb`. They always load our datasets from
   `bdatm-project`, so the models are trained and tested on exactly the same data as ours.

On Colab each notebook clones this repository by itself, from the branch set in
`CONFIG["branch"]`. To run locally, set `CONFIG["colab"]` to `False` in the notebook and install
the packages with `pip install -r requirements.txt`. On Colab, the diagnostics cell (Section 11)
asks for access to Google Drive, where it saves its tables.

Building the datasets again (`Datasets_creator.ipynb`) is not needed: the task notebooks use our
datasets on Hugging Face. To use our exact version of the Lichess database, see [Dataset](#dataset).
To build and upload the datasets anyway, set `ORGANIZATION_NAME` in its upload cell to your own namespace, and change `dataset_name` in the
task notebooks to load your datasets.

If a notebook is stopped, running it again picks up training from the last checkpoint saved on
the Hub and skips the evaluations that are already done, so nothing is computed twice. On an L4
GPU, evaluating one model on the 400 test samples takes about 20-25 minutes for Task 1 and about 2
minutes for Task 2 and Task 3.

**Random seeds.** Seed 42 fixes the dataset sampling and split, and the order of the training data
in both training loops. In the fixed-order LoRA runs it also fixes dropout (set by the Hugging Face
Trainer). The initial LoRA weights were not seeded, nor, in the learned-reordering runs, dropout
and the orders sampled by the policy. Retraining therefore gives close but not identical models;
our published results are checked by evaluating the published models again (see
[How to verify the results](#how-to-verify-the-results)).

## Authors

- Tommaso Bergonzoni
- Ivo Rimbaldi
