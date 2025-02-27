from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from argparse import ArgumentParser
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from calculate_score import edit_distance
from utils import DataSet
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_path)
print(current_path)


class Embedding(object):
    def __init__(self,emb_model_name_or_path,**kwargs):
        self.emb_model_name_or_path = emb_model_name_or_path
        self.device = kwargs.get('device','cuda')
        self.load_emb_model()

    # load model
    def load_emb_model(self):
        self.emb_model = AutoModel.from_pretrained(self.emb_model_name_or_path, trust_remote_code=True).half().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.emb_model_name_or_path, trust_remote_code=True)


def embed_query(text,args,device="cuda"):
    emb_model_name_or_path =args.emb_model_name_or_path
    model = AutoModel.from_pretrained(args.emb_model_name_or_path, trust_remote_code=True).half().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.emb_model_name_or_path, trust_remote_code=True)
    if 'bge' in emb_model_name_or_path:
        DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
    else:
        DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
    text = text.replace("\n", " ")
    if 'bge' in args.emb_model_name_or_path:
        encoded_input = tokenizer([DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True,
                                        truncation=True, return_tensors='pt').to(device)  
    else:
        encoded_input = tokenizer([text], padding=True,
                                        truncation=True, return_tensors='pt').to(device)
  
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
    return sentence_embeddings[0].tolist()


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--llm_model_name_or_path', default='llm/Qwen1.5-7B-Chat', help='the path of llm to use')
    parser.add_argument('--emb_model_name_or_path', default="llm/bge-large-zh", help='the path of embedding model to use')
    parser.add_argument('--rerank_model_name_or_path', default='llm/bge-reranker-large', help='the path of rerank model to use')
    parser.add_argument('--retrieval_methods', default=None, help='the method to retrieval, you can choose from [bm25, emb]')
    parser.add_argument('--corpus_path', default='demo_data/t_knowledge_e0.jsonl', help='corpus path')
    parser.add_argument('--test_query_path', default='demo_data/t_o.jsonl', help='test query path')
    parser.add_argument('--num_input_docs', default=5, help='num of context docs')
    parser.add_argument('--save_path',default="demo_data/t.npy",help="the path to save the predict result")
    parser.add_argument('--data_name',default="t",help="data name")
    parser.add_argument('--num_workers',default=5,type=int,help="workers num")
    return parser.parse_args()       

           
if __name__=="__main__":
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    data_set = DataSet() 
    data = data_set.load_json_data(args.test_query_path)
    npy_list = []
    for d in data:
        text = d['question']
        emb = np.array(embed_query(text,args)[:768]).astype(np.float32)
        npy_list.append(emb)
    npy_list = np.array(npy_list)
    np.save(args.save_path,npy_list)
    print('save emb to: ',args.save_path)
