from pathlib import Path
import fitz


def pdf_to_png(file_name: str) -> None:
    parent_path = Path(__file__).resolve().parent
    file_path = parent_path / f"{file_name}.pdf"
    dpi = 300
    zoom = dpi / 72
    magnify = fitz.Matrix(zoom, zoom)
    doc = fitz.open(filename=file_path)
    for page in doc:
        pix = page.get_pixmap(matrix=magnify)
        pix.save(parent_path / f"{file_name}_{page.number}.png")


pdf_to_png("block_diagram")
