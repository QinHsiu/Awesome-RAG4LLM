

import os
import re
import torch
import torch.nn as nn
import faiss
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool 
import concurrent.futures


class DataSet(object):
    def __init__(self):
        pass

    def load_data(self, data_path):
        with open(data_path,'r+',encoding='utf-8') as fr:
            data = fr.readlines()
        data = [d.strip() for d in data]
        return data

    def load_json_data(self, json_data_path):
        json_data = self.load_data(json_data_path)
        data = [json.loads(d) for d in json_data]
        return data
    
    def load_excel_data(self, excel_data_path, sheet_name='Sheet1'):
        excel_data = pd.read_excel(excel_data_path, sheet_name=sheet_name)
        return excel_data

    
    def dump_json_file(self,save_data,save_path,mode="w+"):
        with open(save_path,mode,encoding='utf-8') as fw:
            fw.write(json.dumps(save_data,ensure_ascii=False)+"\n")


    def cat_json_file(self, json_file_list, save_path):
        json_data_zero = self.load_json_data(json_file_list[0])
        l = len(json_data_zero)
        for json_file in json_file_list[1:]:
            json_data = self.load_json_data(json_file)
            for i in range(l):
                for k in json_data[i]:
                    json_data_zero[i][k] = json_data[i][k]
        for i in range(l):
            # print(json_data_zero[i])
            self.dump_json_file(json_data_zero[i], save_path, 'a+')

    def add_column_by2json(self, dp1, dp2, merge_dic={'content_is_same':'question_is_same'},sp='1.jsonl'):
        data1 = self.load_json_data(dp1)
        data2 = self.load_json_data(dp2)
        for i,d in enumerate(zip(data1,data2)):
            d1, d2 = d[0], d[1]
            for k_ in merge_dic:
                d1[k_] = d2['__dj__stats__'][merge_dic[k_]]
            self.dump_json_file(d1,sp,'a+')
            
    def add_column_by2jsonv1(self, dp1, dp2, merge_key='cosin_sim',sp='1.jsonl'):
        data1 = self.load_json_data(dp1)
        data2 = self.load_json_data(dp2)
        for i,d in enumerate(zip(data1,data2)):
            d1, d2 = d[0], d[1]
            d1[merge_key] = d2[merge_key]
            self.dump_json_file(d1,sp,'a+')
    

    def clean_string(self,value):
        # 移除 LaTeX 表达式
        value = re.sub(r'\$\$.*?\$\$', '', value)
        # 移除非法字符
        illegal_chars = ['\\', '/', '*', '[', ']', ':', '?']
        for char in illegal_chars:
            value = value.replace(char, "")
        return value
    
    def clean_string_v1(self,value):
        return re.sub(r'[\n,\t,' ']', '', value)
    
    def clean_string(self,s):
        res = ''
        for i in range(len(s)):
            if s[i] == '\\': #in ['\\','\f','\t']:
                t = s[i].replace('\\','\\\\')
            elif s[i] == '\f':
                t = s[i].replace('\f','\\f')
            elif s[i] == '\t':
                t = s[i].replace('\t','\\t')
            else:
                t = s[i]
            res  += t

        return res
                

    def change_json_data2excel(self, json_dp, excel_save_path):
        data = self.load_json_data(json_dp)
        pd_save={d:[] for d in data[0].keys()}
        # print(pd_save)
        for i,d in enumerate(data):
            for k in pd_save:
                pd_save[k].append(self.clean_string(r'{}'.format(str(d[k]))))
        pd_save = pd.DataFrame(pd_save)
        pd_save.to_excel(excel_save_path,index=False,engine='openpyxl')
        
        
    
            
    def change_excel_data2json(self, excel_data_path, json_save_path, sheet_name='Sheet1'):
        excel_data=pd.read_excel(excel_data_path,sheet_name=sheet_name)
        key_list=list(excel_data.keys())
        print("key_list: ",key_list)
        print("") 
        for idx in range(excel_data.shape[0]):
            s={}
            for k in key_list:
                if 'Unnamed' in k:
                    continue
                t=excel_data.loc[idx,k]
                s[k]=r"{}".format(t)
            self.dump_json_file(s,json_save_path,"a+")
        print("Done!")

if __name__ == '__main__':
    pass