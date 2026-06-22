import pdfplumber

with pdfplumber.open("test_doc.pdf") as pdf:
    # 先看第35页(对应你之前看到的 page_label: '35',索引是34)
    page = pdf.pages[34]
    text = page.extract_text()
    print("=== pdfplumber 提取的第35页内容 ===")
    print(text)
    print(f"\n字符数: {len(text) if text else 0}")