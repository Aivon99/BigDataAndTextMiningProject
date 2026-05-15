import chess
import chess.svg
import cairosvg
import io
from PIL import Image

class ChessRenderer:
    def __init__(self, size: int = 350):
        self.size = size

    def fen_to_image(self, fen: str, lastmove_uci: str = None) -> Image.Image:
        """
        Converte una stringa FEN in un'immagine PNG. 
        Se viene passato lastmove_uci, applica l'highlight di quella mossa.
        """
        board = chess.Board(fen)
        lastmove = chess.Move.from_uci(lastmove_uci) if lastmove_uci else None
        
        # Genera SVG tramite python-chess
        svg_data = chess.svg.board(
            board=board, 
            size=self.size, 
            lastmove=lastmove
        )
        
        # Converte SVG in PNG usando cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        
        # Ritorna un oggetto PIL Image (richiesto da HuggingFace Datasets)
        return Image.open(io.BytesIO(png_data)).convert("RGB")

    def fen_to_ascii(self, fen: str) -> str:
        """
        Genera la rappresentazione ASCII della scacchiera come richiesto dal prof.
        """
        board = chess.Board(fen)
        ascii_str = ""
        
        # Rank da 8 a 1 (indici da 7 a 0)
        for rank in range(7, -1, -1):
            row_str = []
            # File da 'a' ad 'h' (indici da 0 a 7)
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                if piece:
                    row_str.append(piece.symbol())
                else:
                    row_str.append('.')
            ascii_str += " ".join(row_str) + "\n"
            
        return ascii_str.strip()