import os
import fitz
import pdfplumber
from time import time
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from calculate_score import edit_distance
current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)

class PDFReader(object):
    def __init__(self):
        pass

    def gen_pdf_res(self,dp,mode='pymupdf'):
        if mode=='pymupdf':
            return self.extract_pdf_page_text_pymupdf(dp)
        elif mode=='pdfplumber':
            return self.extract_pdf_page_text_pdfplumber(dp)
        elif mode=='PyPDF2':
            return self.extract_pdf_page_text_PyPDF2(dp)
        elif mode=='pdfminer':
            return self.extract_pdf_page_text_pdfminer(dp)
        else:
            raise ValueError('mode not supported')
    
    def extract_pdf_page_text_pymupdf(self, filepath):
        doc = fitz.open(filepath)
        texts = [] 
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            texts.append(text)
        return texts    

    def extract_pdf_page_text_pdfplumber(self, filepath):
        texts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                texts.append(text)
        return texts

    def extract_pdf_page_text_PyPDF2(self, filepath):
        texts = []
        with open(filepath, 'rb') as f:
            pdf = PdfReader(f)
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                text = page.extract_text()
                texts.append(text)
        return texts

    def extract_pdf_page_text_pdfminer(self, filepath):
        texts = []
        with open(filepath, 'rb') as f:
            pdf = extract_text(f)
            texts.append(pdf)
        return texts

    def cal_edit_distance(self, str1, str2):
        return edit_distance(str1, str2)

    def cal_time_cost(self, func, *args):
        t_b = time()
        res = func(*args)
        t_e = time()
        if isinstance(res, list):
            res = ''.join(res)
        return res, t_e-t_b

    def judge_res(self,ori_str,pdf_path):
        modes = ['pymupdf','pdfplumber','PyPDF2','pdfminer']
        for mode in modes:
            res,t = self.cal_time_cost(self.gen_pdf_res,pdf_path,mode)
            edit_dis = self.cal_edit_distance(ori_str,res)
            print('mode:',mode)
            # print('res:',res)
            print('edit_dis:',edit_dis)
            print('time cost:',t)
            print('')
          
# mode: pymupdf
# edit_dis: 21
# time cost: 0.003134012222290039

# mode: pdfplumber
# edit_dis: 20
# time cost: 0.03606843948364258

# mode: PyPDF2
# edit_dis: 20
# time cost: 0.010962247848510742

# mode: pdfminer
# edit_dis: 30
# time cost: 0.02911233901977539


if __name__ == '__main__':
    pdf_reader = PDFReader()
    ori_str = '''Demo\n要跑赢通货膨胀，年化收益率需要超过通货膨胀率。具体的目标年化收益率取决于当前和预期的通货膨胀率。\n一般来说，如果通货膨胀率是3%，那么你的投资回报率至少应该达到3%以上才能保值。如果你希望不仅仅是保值，还想获得实际收益（即购买力增加），则需要在此基础上再加上一些额外的收益。例如，你可能希望实现5%或更高的年化收益率，以确保在扣除通货膨胀后的净收益为正数。\n\n请记住，投资涉及风险，不同类型的投资产品有不同的风险和潜在回报。在制定投资策略时，建议综合考虑个人的风险承受能力、财务目标和市场状况，并咨询专业的金融顾问以做出明智的决定。'''
    pdf_path = os.path.join(current_path, '../demo_data/t.pdf')
    pdf_reader.judge_res(ori_str,pdf_path)
 
      


        
