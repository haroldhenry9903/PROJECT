import fitz

def extract_pdf():
    doc = fitz.open('FYP1-FullReport-82580(updatedv1).pdf')
    text = ""
    for page in doc:
        text += page.get_text()
    
    with open('report_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    print("Extraction complete. Text saved to report_text.txt")

if __name__ == '__main__':
    extract_pdf()
