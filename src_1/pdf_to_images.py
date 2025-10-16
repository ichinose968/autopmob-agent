from typing import List, Tuple
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF

def pdf_to_png_bytes(path: str, dpi: int = 220) -> List[Tuple[int, bytes]]:
    """
    Returns list of (page_index, png_bytes)
    """
    doc = fitz.open(path)
    out = []
    for i, page in enumerate(doc):
        # レンダリング（解像度はトークンコストと品質の妥協点を調整）
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = BytesIO()
        img.save(buf, format="PNG")
        out.append((i, buf.getvalue()))
    return out
