import re
import fitz
import json
import pandas as pd
from tqdm import tqdm
from utils import pre_process

class Reader:
    def __init__(self, corpus_path: str):
        self.corpus = None
        # PDF file
        if corpus_path.endswith('.pdf'):
            self.corpus = self.extract_pdf_page_text_pymupdf(corpus_path)
        # CSV file
        elif 'Multi-CPR' in corpus_path:
            self.corpus = self.extract_multiCPR_text(corpus_path)
        # JSON file
        elif '.json' in corpus_path:
            self.corpus = self.extract_json_file(corpus_path)
        # Other file
        else:
            self.corpus = self.extract_other_file(corpus_path)
        
    def extract_pdf_page_text_pymupdf(self, filepath, max_len=256, overlap_len=100):
        doc = fitz.open(filepath)
        texts = [] 
        for page_num,_ in enumerate(tqdm(doc,desc='解析PDF文件......')):
            page = doc.load_page(page_num)
            page_text = page.get_text().strip()
            text = pre_process(page_text)  
            texts.append(text)
        # merge all texts
        all_page_texts = ''.join(texts)
        all_page_texts = all_page_texts.replace('\n','')
        total_length = len(all_page_texts)
        texts = [all_page_texts[i:i+max_len] for i in range(0,total_length,overlap_len)]
        return texts    


    def extract_pdf_page_text(self, filepath, max_len=256, overlap_len=100):
        """Reference from """
        page_content  = []
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in tqdm.tqdm(pdf_reader.pages, desc='解析PDF文件...'):
                page_text = page.extract_text().strip()
                raw_text = [text.strip() for text in page_text.split('\n')]
                new_text = '\n'.join(raw_text)
                new_text = re.sub(r'\n\d{2,3}\s?', '\n', new_text)
                if len(new_text) > 10 and '..............' not in new_text:
                    page_content.append(new_text)
        cleaned_chunks = []
        i = 0
        # 暴力将整个pdf当做一个字符串，然后按照固定大小的滑动窗口切割
        all_str = ''.join(page_content)
        all_str = all_str.replace('\n', '')
        while i < len(all_str):
            cur_s = all_str[i:i+max_len]
            if len(cur_s) > 10:
                cleaned_chunks.append(cur_s)
            i += (max_len - overlap_len)
        return cleaned_chunks

    def extract_multiCPR_text(self, filepath):
        corpus = pd.read_csv(filepath, sep='\t', header=None)
        corpus.columns = ['pid', 'passage']
        return corpus.passage.values.tolist()


    def extract_json_file(self, filepath):
        with open(filepath,"rb") as fr:
            data=fr.readlines()
        chunks=[]
        for idx,line in enumerate(tqdm(data)):
            line=json.loads(line.strip())
            chunks.append(line)
        return chunks

    def extract_other_file(self, file_path):
        pass
            
            
            
        
  