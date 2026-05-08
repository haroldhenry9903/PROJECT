import fitz
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent


def extract_pdf():
    doc = fitz.open(_TOOLS_DIR / "FYP1-FullReport-82580(updatedv1).pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    out_path = _TOOLS_DIR / "report_text.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Extraction complete. Text saved to {out_path}")


if __name__ == "__main__":
    extract_pdf()
