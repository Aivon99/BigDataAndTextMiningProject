from io import BytesIO

import chess
import chess.svg
import cairosvg

from PIL import Image


def fen_to_board(fen: str) -> chess.Board:
    """
    Convert FEN string into python-chess Board.
    """

    return chess.Board(fen)


def create_svg_board(
    board: chess.Board,
    size: int = 512,
    lastmove=None,
):
    """
    Generate SVG chessboard.
    """

    return chess.svg.board(
        board=board,
        size=size,
        lastmove=lastmove
    )


def svg_to_pil(svg_string: str) -> Image.Image:
    """
    Convert SVG string to PIL image.
    """

    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8")
    )

    image = Image.open(
        BytesIO(png_bytes)
    ).convert("RGB")

    return image


def render_board(
    fen: str,
    size: int = 512,
    lastmove=None
) -> Image.Image:
    """
    Complete rendering pipeline:
    FEN -> SVG -> PIL.
    """

    board = fen_to_board(fen)

    svg = create_svg_board(
        board=board,
        size=size,
        lastmove=lastmove
    )

    return svg_to_pil(svg)