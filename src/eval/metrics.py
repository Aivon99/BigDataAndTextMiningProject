"""Metrics used to score the FEN and SAN predictions."""

import Levenshtein
import chess
import re


_FEN_RE = re.compile(
    r"(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+"
    r"(?: +[wb] +(?:-|[KQkq]+) +(?:-|[a-h][36]) +[0-9]+ +[0-9]+)?"
)


def extract_fen(text):
    """Returns the first FEN found in the model output, or the whole output if
    there is none.
    """
    match = _FEN_RE.search(text)
    return match.group(0).strip() if match else text.strip()


def calculate_fen_exact_match(predicted_fen, ground_truth_fen):
    """1.0 if the whole FEN string is identical, else 0.0."""
    return 1.0 if predicted_fen.strip() == ground_truth_fen.strip() else 0.0


def calculate_levenshtein_metrics(predicted_fen, ground_truth_fen):
    """Edit distance between the two FEN strings, and the same distance divided
    by the length of the ground truth (character error rate).
    """
    pred = predicted_fen.strip()
    gt = ground_truth_fen.strip()

    dist = Levenshtein.distance(pred, gt)
    cer = dist / max(len(gt), 1)

    return {
        "levenshtein_distance": dist,
        "character_error_rate": cer
    }


def calculate_square_by_square_accuracy(predicted_fen, ground_truth_fen):
    """Share of the 64 squares that match. Returns 0 if the predicted FEN
    can't be parsed or is empty.
    """
    try:
        pred_board = chess.Board(predicted_fen.strip().split()[0])
        gt_board = chess.Board(ground_truth_fen.strip().split()[0])
    except (ValueError, IndexError):
        return 0.0

    correct_squares = 0
    total_squares = 64

    for square in chess.SQUARES:
        pred_piece = pred_board.piece_at(square)
        gt_piece = gt_board.piece_at(square)
        if pred_piece == gt_piece:
            correct_squares += 1

    return correct_squares / total_squares

def calculate_san_exact_match(predicted_move, ground_truth_move):
    """1 if the raw answer is exactly the SAN move, else 0."""
    return 1 if predicted_move.strip() == ground_truth_move.strip() else 0
