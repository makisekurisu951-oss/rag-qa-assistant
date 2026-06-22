import pdfplumber

with pdfplumber.open("test_doc.pdf") as pdf:
    print(f"文档总页数: {len(pdf.pages)}\n")
    
    # 遍历所有页,找出文字最多的几页(大概率是正文内容页)
    page_lengths = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        length = len(text) if text else 0
        page_lengths.append((i, length))
    
    # 按内容长度排序,看前5页最长的
    page_lengths.sort(key=lambda x: x[1], reverse=True)
    print("内容最多的5页:")
    for idx, length in page_lengths[:5]:
        print(f"  第{idx}页(page_label可能不同),字符数: {length}")
    
    # 打印内容最多那一页的具体文字
    top_page_idx = page_lengths[0][0]
    print(f"\n=== 第{top_page_idx}页完整内容 ===")
    print(pdf.pages[top_page_idx].extract_text())