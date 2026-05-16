FEN_PROMPT = """
You are a specialized model for chessboard understanding.

Your task is to generate the FEN representation
of the given chessboard image.

Return ONLY the FEN string.
"""

ASCII_PROMPT = """
You are a specialized chessboard reasoning model.

Generate the ASCII representation
of the chessboard.

Rules:
- 8 ranks from rank 8 to rank 1
- files a-h left to right
- uppercase = white pieces
- lowercase = black pieces
- '.' = empty square

Return ONLY the ASCII board.
"""

SAN_PROMPT = """
You are a specialized chess reasoning model.

Given the highlighted move on the board,
generate the SAN notation
of the move.

Return ONLY the SAN move.
"""