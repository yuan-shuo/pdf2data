import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    # 移除换行符
    text = text.replace("\n", "")
    return text

# 替换为你的PDF文件路径
pdf_path = "pdfSQL/topdf.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text)
