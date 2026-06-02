import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_DIR = _TOOLS_DIR.parent

def extract_docx():
    docx_path = _REPO_DIR / "FYP2-FinalReport(DRAFT)-82580.docx"
    out_path = _TOOLS_DIR / "fyp2_report_text.txt"
    
    print(f"Extracting {docx_path} to {out_path}...")
    
    # Try importing docx (python-docx) first
    try:
        import docx
        doc = docx.Document(docx_path)
        text = []
        
        # In python-docx, document.paragraphs only contains top-level paragraphs, not paragraphs in tables.
        # But we can iterate over body elements to keep them in order.
        # Let's just extract paragraphs and then tables for now, or use a custom block iterator.
        
        # Custom block iterator to preserve order
        for block in doc.element.body:
            if block.tag.endswith('p'):
                p = docx.text.paragraph.Paragraph(block, doc)
                text.append(p.text)
            elif block.tag.endswith('tbl'):
                t = docx.table.Table(block, doc)
                text.append("\n--- TABLE START ---")
                for row in t.rows:
                    cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
                    text.append(" | ".join(cells))
                text.append("--- TABLE END ---\n")
                
        full_text = "\n".join(text)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Extraction via python-docx complete. Saved to {out_path}")
        return
    except Exception as e:
        print(f"python-docx failed or not found ({e}), using zipfile XML parsing fallback...")
        
    # XML fallback that preserves order of tables and paragraphs
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    def get_text_from_p(p_elem):
        texts = []
        for text_elem in p_elem.findall('.//w:t', ns):
            if text_elem.text:
                texts.append(text_elem.text)
        return "".join(texts)
        
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Find body element
            body = root.find('.//w:body', ns)
            if body is None:
                # Fallback to searching all elements
                body = root
                
            text = []
            for child in body:
                if child.tag.endswith('p'):
                    text.append(get_text_from_p(child))
                elif child.tag.endswith('tbl'):
                    text.append("\n--- TABLE START ---")
                    for row in child.findall('.//w:tr', ns):
                        cells_text = []
                        for cell in row.findall('.//w:tc', ns):
                            # Cell contains one or more paragraphs
                            cell_ps = []
                            for p in cell.findall('.//w:p', ns):
                                cell_ps.append(get_text_from_p(p))
                            cells_text.append(" ".join(cell_ps).replace("\n", " ").strip())
                        text.append(" | ".join(cells_text))
                    text.append("--- TABLE END ---\n")
                    
            full_text = "\n".join(text)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"Extraction via XML fallback complete. Saved to {out_path}")
    except Exception as ex:
        print(f"XML fallback failed: {ex}")

if __name__ == "__main__":
    extract_docx()
