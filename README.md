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
[Legacy runs](#legacy-runs)). Their results are kept in the results tables, labelled
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

Lichess publishes only the latest version of the puzzle database and replaces it regularly, without
keeping the older versions, so running `Datasets_creator.ipynb` today downloads a newer file and
gives a different sample. The exact datasets we used are on Hugging Face (see the links above), so
all the experiments can be reproduced from them.

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
  the policy samples orders, and training runs for 4 epochs in total. The REOrder code is
  downloaded automatically into `external/REOrder` (at a fixed commit) the first time it is needed.

LoRA settings: r = 16, alpha = 32, dropout 0.05 on all attention and MLP projections,
learning rate 2e-4, batch size 8, up to 10 epochs with early stopping on the validation loss
(the best epoch is kept).

Metrics:

- Task 1: FEN exact match, character error rate, Levenshtein distance and square-by-square
  accuracy, plus separate checks for each part of the FEN. The main score is the exact match on
  the board part of the FEN: side to move, castling rights and move counters cannot be seen in
  the image, so the full-string exact match is only 1-3% for every model.
- Task 2 and Task 3: exact match on the SAN move, plus extra checks such as whether the
  predicted move is legal or is the same move written in a different way.

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
  accuracy is above 97.9% for every order.
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
image is not changed, so that comparison is fair.

## Repository structure

```
configs/
  config.yaml              settings for the dataset generation
notebooks/
  Datasets_creator.ipynb   builds the three datasets and uploads them to Hugging Face
  Task_1.ipynb             training and evaluation for Task 1
  Task_2.ipynb             training and evaluation for Task 2
  Task_3.ipynb             training and evaluation for Task 3
src/
  data/
    generation.py          draws the boards and builds the samples
    utilities.py           loads the config and the Lichess CSV
  eval/
    metrics.py             FEN and SAN metrics
    evaluation.py          runs a model on the test set of each task
    results.py             saves and loads the results on Hugging Face
    diagnostics.py         extra analysis of the errors and of the training runs
    legacy/                notebook of an early Task 2 run with pixel-tile reordering
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

## How to run

The notebooks are meant to run on Google Colab with a GPU, but they also work locally.

1. Add a Hugging Face token with write access to the `bdatm-project` organization: on Colab as a
   secret called `HF_TOKEN`, locally in a `.env` file in the project folder.
2. Run the notebooks in this order:
   1. `Datasets_creator.ipynb` (only needed to build the datasets again)
   2. `Task_1.ipynb`
   3. `Task_2.ipynb`
   4. `Task_3.ipynb`

On Colab each notebook clones this repository by itself, from the branch set in
`CONFIG["branch"]`. To run locally, set `CONFIG["colab"]` to `False` in the notebook and install
the packages with `pip install -r requirements.txt`. The task notebooks also install
`flash-linear-attention`.

The last cell of `Task_1.ipynb` saves the notebook to GitHub and closes the Colab runtime. It only
runs on Colab (`CONFIG["colab"] = True`) and needs a GitHub token saved as a Colab secret called
`GITHUB_TOKEN_BIG`.

Trained models and results are saved on Hugging Face as the notebooks run. If a notebook is
stopped, running it again picks up training from the last checkpoint and skips the evaluations
that are already done, so nothing is computed twice.

All the repositories are under the `bdatm-project` organization, set by `hf_org_prefix` in the
notebooks. To retrain the models from scratch, change `hf_org_prefix` to your own Hugging Face
namespace.

## Authors

- Tommaso Bergonzoni
- Ivo Rimbaldi
