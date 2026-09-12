from typing import List, Optional

import chess


def levenshtein_distance(a: str, b: str) -> int:
    """
    Classic edit-distance DP. Pure Python is fine here: FEN/SAN strings are
    short (well under 100 chars), so no need for a compiled dependency.
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,      # deletion
                curr_row[j - 1] + 1,  # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row

    return prev_row[-1]


def character_error_rate(pred: str, target: str) -> float:
    """
    Levenshtein distance normalized by target length. 0.0 = perfect match.
    """
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, target) / len(target)


def _fen_placement_field(fen: str) -> str:
    """
    Isolates the piece-placement field (first field) of a FEN string.
    """
    return fen.strip().split(" ")[0]


def fen_exact_match(pred_fen: str, target_fen: str, placement_only: bool = True) -> bool:
    """
    Compares two FEN strings. Defaults to comparing only the piece-placement
    field: side-to-move, castling rights, en passant target, halfmove clock,
    and fullmove number are not visually recoverable from a static board
    image, so scoring the full 6-field FEN would penalize a model for
    information it was never shown. Set placement_only=False to require a
    full-string match instead.
    """
    if placement_only:
        return _fen_placement_field(pred_fen) == _fen_placement_field(target_fen)
    return pred_fen.strip() == target_fen.strip()


def _fen_placement_to_squares(fen: str) -> Optional[List[str]]:
    """
    Parses a FEN's piece-placement field into a 64-length list of piece
    symbols ('.' for empty), ordered a1..h8 (matching `chess.SQUARES`).
    Returns None if the placement field isn't a well-formed 8x8 layout.
    """
    placement = _fen_placement_field(fen)
    try:
        board = chess.Board(f"{placement} w - - 0 1")
    except ValueError:
        return None
    return [
        (board.piece_at(square).symbol() if board.piece_at(square) else ".")
        for square in chess.SQUARES
    ]


def square_by_square_accuracy(pred_fen: str, target_fen: str) -> float:
    """
    Fraction of the 64 squares where the predicted piece placement matches
    the target. Returns 0.0 if the predicted FEN can't be parsed at all.
    """
    target_squares = _fen_placement_to_squares(target_fen)
    if target_squares is None:
        raise ValueError(f"Invalid target FEN: {target_fen}")

    pred_squares = _fen_placement_to_squares(pred_fen)
    if pred_squares is None:
        return 0.0

    matches = sum(p == t for p, t in zip(pred_squares, target_squares))
    return matches / len(target_squares)


def san_exact_match(pred_san: str, target_san: str) -> bool:
    """
    Exact-match on Standard Algebraic Notation, ignoring surrounding
    whitespace only (SAN is compact enough that any other difference is a
    real disagreement, e.g. missing a '+'/'#' suffix or a disambiguator).
    """
    return pred_san.strip() == target_san.strip()
