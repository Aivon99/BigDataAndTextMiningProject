# Chessboard Understanding with a Vision-Language Model

Project for the **Big Data Analytics and Text Mining** course at the **University of Bologna**.

- Professor: Gianluca Moro, University of Bologna
- Tutor: Ing. Giacomo Frisoni

We fine-tune a small vision-language model (Qwen3.5-0.8B) with LoRA to read chessboard images,
and we study how the order of the image patches changes the results, following the idea of the
REOrder paper.

## Hugging Face

Datasets, trained models and results are stored in the
[bdatm-project](https://huggingface.co/bdatm-project) organization on Hugging Face.

| What | Repositories |
|---|---|
| Datasets | [dataset_task1](https://huggingface.co/datasets/bdatm-project/dataset_task1), [dataset_task2](https://huggingface.co/datasets/bdatm-project/dataset_task2), [dataset_task3](https://huggingface.co/datasets/bdatm-project/dataset_task3) |
| LoRA models | `qwen-task{1,2,3}-{strategy}-lora`, for example [qwen-task1-standard-lora](https://huggingface.co/bdatm-project/qwen-task1-standard-lora) |
| Results | [evaluation-results-task1](https://huggingface.co/datasets/bdatm-project/evaluation-results-task1), [evaluation-results-task2](https://huggingface.co/datasets/bdatm-project/evaluation-results-task2), [evaluation-results-task3](https://huggingface.co/datasets/bdatm-project/evaluation-results-task3) |

The strategy in the model name is `standard`, `zigzag`, `spiral`, `file_wise` or
`learned-reordering`.

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
- The puzzles are split 80/10/10 into train, validation and test (3,200 / 400 / 400).
- There is one dataset for each task, and all three are built from the same 4,000 puzzles.
  Only the images and the expected answer change from one task to another.
- Boards are drawn as 512x512 images without coordinates, so every square lines up with the
  patch grid of the model.

These numbers can be changed in `configs/config.yaml`. The datasets are already on Hugging Face
(see the links above).

## Method

For each task we compare:

- **Vanilla**: the base model with no training (zero-shot).
- **LoRA**: the model fine-tuned with LoRA on the normal images (raster order).
- **Patch reordering, training-free (TF)**: the image is cut into an 8x8 grid of tiles that are
  put back in a different order (zigzag, spiral or file_wise). The LoRA model above is tested on
  these images without any new training.
- **Patch reordering, fine-tuned (SFT)**: a new LoRA model is trained directly on the reordered
  images, one for each strategy.
- **Learned reordering**: a small policy (Plackett-Luce) learns which tile order works best,
  trained together with LoRA using REINFORCE.

Metrics:

- Task 1: FEN exact match, character error rate, Levenshtein distance and square-by-square
  accuracy, plus separate checks for each part of the FEN.
- Task 2 and Task 3: exact match on the SAN move, plus extra checks such as whether the
  predicted move is legal or is the same move written in a different way.

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
  training/
    checkpoints.py         saves and resumes training runs on Hugging Face
    preprocessing.py       turns dataset rows into model inputs
    reordering.py          the patch reordering strategies
    finetune.py            LoRA fine-tuning for one task and one strategy
    learned_reordering.py  learned reordering with a Plackett-Luce policy
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

On Colab each notebook clones this repository by itself. To run locally, set
`CONFIG["colab"]` to `False` in the notebook and install the packages with
`pip install -r requirements.txt`.

Trained models and results are saved on Hugging Face as the notebooks run. If a notebook is
stopped, running it again picks up training from the last checkpoint and skips the evaluations
that are already done, so nothing is computed twice.

## Results

The results of all models are on Hugging Face:

- [Task 1 results](https://huggingface.co/datasets/bdatm-project/evaluation-results-task1)
- [Task 2 results](https://huggingface.co/datasets/bdatm-project/evaluation-results-task2)
- [Task 3 results](https://huggingface.co/datasets/bdatm-project/evaluation-results-task3)

The trained LoRA models are in the [bdatm-project](https://huggingface.co/bdatm-project)
organization.

## Authors

- Tommaso Bergonzoni
- Ivo Rimbaldi
