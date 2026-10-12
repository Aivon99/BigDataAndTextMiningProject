import Levenshtein
import chess
import re


_FEN_RE = re.compile(
    r"(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+"
    r"(?: +[wb] +(?:-|[KQkq]+) +(?:-|[a-h][36]) +[0-9]+ +[0-9]+)?"
)


def extract_fen(text: str) -> str:
    """
    Pulls the first FEN-shaped string out of a model reply (board part plus, if
    present, the side/castling/en-passant/clock fields). A chatty answer like
    "The FEN is r3k2r/... w KQkq - 1 13." is scored on the FEN it contains
    instead of on all the surrounding words; if nothing FEN-shaped is found the
    stripped reply is returned unchanged, so garbage is still scored as garbage.
    """
    match = _FEN_RE.search(text)
    return match.group(0).strip() if match else text.strip()


def calculate_fen_exact_match(predicted_fen: str, ground_truth_fen: str) -> float:
    """
    Calculates FEN Exact Match: returns 1.0 if the predicted FEN string 
    matches the ground truth exactly, otherwise 0.0.
    """
    return 1.0 if predicted_fen.strip() == ground_truth_fen.strip() else 0.0


def calculate_levenshtein_metrics(predicted_fen: str, ground_truth_fen: str) -> dict:
    """
    Calculates the Levenshtein distance and Character Error Rate (CER) 
    between the predicted FEN and the ground truth FEN string.
    """
    pred = predicted_fen.strip()
    gt = ground_truth_fen.strip()
    
    dist = Levenshtein.distance(pred, gt)
    cer = dist / max(len(gt), 1)
    
    return {
        "levenshtein_distance": dist,
        "character_error_rate": cer
    }


def calculate_square_by_square_accuracy(predicted_fen: str, ground_truth_fen: str) -> float:
    """
    Calculates Square-by-Square Accuracy across the 64 squares of the chessboard.
    It parses both FEN strings into python-chess Board objects and compares 
    each of the 64 squares individually.
    """
    try:
        # We only consider the piece placement part of the FEN string (before the first space)
        pred_board = chess.Board(predicted_fen.strip().split()[0])
        gt_board = chess.Board(ground_truth_fen.strip().split()[0])
    except ValueError:
        # If the predicted FEN is malformed and cannot be parsed, accuracy is 0.0
        return 0.0
    
    correct_squares = 0
    total_squares = 64
    
    for square in chess.SQUARES:
        pred_piece = pred_board.piece_at(square)
        gt_piece = gt_board.piece_at(square)
        if pred_piece == gt_piece:
            correct_squares += 1
            
    return correct_squares / total_squares

def calculate_san_exact_match(predicted_move: str, ground_truth_move: str) -> int:
    """
    Computes Exact Match (EM) for chess moves in SAN notation.
    Returns 1 if predictions match the ground truth exactly, 0 otherwise.
    """
    return 1 if predicted_move.strip() == ground_truth_move.strip() else 0
