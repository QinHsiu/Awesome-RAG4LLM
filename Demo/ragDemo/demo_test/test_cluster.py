import os
import json
import torch
import numpy as np
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import pickle
import faiss
import os
from time import time
from multiprocessing import Pool 
import concurrent.futures

# load data
def load_data(jpath):
    t = np.load(jpath)
    return np.load(jpath)[:,:768]

def read_npy(npypath):
    npy_list = []
    num_processes = 2
    pool = Pool(num_processes)
    npy_list = pool.map(load_data, [npypath])
    pool.close()
    pool.join()
    return np.vstack(npy_list)


class QueryBank():
    def __init__(self, *args, **kwargs):
        self.cluster = {}
        self.dim = kwargs.get('emb_dim',768)
        self.query_key = kwargs.get('query_key', 'question')
        self.answer_key = kwargs.get('answer_key', 'model_predict')
        self.answer_weight = kwargs.get('answer_weight', 1.0)
        self.base_index_type = kwargs.get('base_index_type', 'index_flatl2')
        self.device_id = kwargs.get('gpu_id',0)
        self.batch_size = kwargs.get('batch_size',2)
        self.__init_base_index(self.base_index_type)
        
    # choose the base index
    def __init_base_index(self, base_index_type):
        """
            input:
                base_index_type, str: the index type
            output:
                none
        """
        if base_index_type == 'index_flatl2':
            base_index = faiss.IndexFlatL2(self.dim)
        elif base_index_type == 'index_flatip':
            base_index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError('base index error!') 
        self.index = faiss.IndexIDMap(base_index)
    
    def load_data(self, data_path, npy_path):
        """
            input:
                data_path, str: the path to save all contents dic.
                npy_path, str, the embedding path of all content.
            output:
                data, list(str): all the read data of the input files, 
                query, list(ndarray): all content embedding of the input data
        """
        with open(data_path,'r+') as fr:
            data = fr.readlines()
        data = [json.loads(d.strip()) for d in data]
        embed_data = read_npy(npy_path)
        query = torch.from_numpy(embed_data.astype(np.float16)).to("cuda:"+str(self.device_id))
        query = [query[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32) for i in range(len(data))]
        return data, query


    def _init_db(self, npy_path, data_path, topk=1, threshold=1.0):
        """
            input:
                npy_path, str: the directory to save all contents embedding,
                data_path, str: the path to save all contents dic,
                topk, int: the search topk res will be updated,
                threshold, float: the threshold for clustering.
            output:
                None
        """
        data, query = self.load_data(data_path, npy_path)
        data_length = len(data)
        batch_nums = data_length//self.batch_size 
        for b_idx in range(batch_nums):
            b_ = b_idx * self.batch_size
            # print(query[b_],type(query[b_]),query[b_].shape)
            e_ = (b_idx + 1) * self.batch_size
            batch_data = data[b_:e_]
            batch_query_embeds = query[b_:e_]
            batch_query_embeds = np.vstack(batch_query_embeds)
            self.insert_and_update(batch_query_embeds, batch_data, topk ,threshold)
        # cost test
        # time_b = time()
        # update question_bank
        # for i, d in enumerate(tqdm(data)):
        #     self.insert_and_update(query[i], d, topk ,threshold)
            # if (i+1)%10000 == 0:
                # time_e = time()
                # print('process {} nums data cost {}'.format(i, time_e-time_b))
    
    # filter some invalid content before add it into db
    def content_is_invalid_filter(self, query_content):
        """
            input:
                query_content, str: need to be scored.
            output:
                the result after filtering
        """
        if query_content in ['','\n','\n\n','None','nan',None]:
            return True
        return query_content

    # filter some invalid answer beforre add it into db
    def answer_is_invalid_filter(self, answer):
        """
            input:
                answer, str: the answer need to be scored or filter.
            output:
                the result after filtering
        """
        if answer in ['','\n','\n\n','None','nan',None]:
            return True
        return answer

    # insert answers
    def insert_answers(self, query_emb, answers, topk=1, threshold=1.0):
        """
            input:
                query_emb, ndarray: the embedding of the query content,
                answers, dic: all answers,
                topk, int: the search topk res will be updated,
                threhold, float: the threshold for clustering.
            output:
                non_searc_res_cnt, int: non_searc_res_cnt.          

        """
        batch_search_res = self.db_search(query_emb, topk, threshold)
        non_search_res_cnt = 0
        for b_idx, b_a in enumerate(answers): 
            search_res = batch_search_res[b_idx]
            if search_res:
                for centroid in search_res:
                    ori_answers = self.cluster[centroid]['answers']
                    for new_a in b_a:
                        if new_a not in ori_answers:
                            ori_answers[new_a] = [0,''] 
                        ori_answers[new_a][0] += 1
                    self.cluster[centroid]['answers'] = ori_answers
            else:
                non_search_res_cnt += 1
        return non_search_res_cnt
                
    def insert_and_update(self, query_emb, data, topk=1, threshold=0.9):
        """
            input:
                query_emb, ndarray: the embedding of the query content,
                data, dic: one sample data, include query_key and answer_key,
                topk, int: the search topk res will be updated,
                threhold, float: the threshold for clustering.
            output:
                None
        """
        batch_search_res = self.db_search(query_emb, topk, threshold)
        for b_idx, d in enumerate(data): 
            query_content = d[self.query_key]
            answer = d[self.answer_key]
            if not self.content_is_invalid_filter(query_content):
                return 
            search_res = batch_search_res[b_idx]
            query_emb_t = query_emb[b_idx,:].reshape(1,-1)
            if search_res:
                self.db_update(search_res, query_emb_t, query_content, answer)
            else:
                self.db_insert(query_emb_t, query_content, answer)

    def cal_sim_score(self,array1, array2):
        """
            input:
                array1, ndarray: for calculate cosin similarity,
                array2, ndarray: for calculate cosin similarity.
            output:
                the cosin similarity about two array. 
        """
        flat_array1 = array1.flatten()
        flat_array2 = array2.flatten()
        dot_product = np.dot(flat_array1, flat_array2)
        norm_array1 = np.linalg.norm(flat_array1)
        norm_array2 = np.linalg.norm(flat_array2)
        similarity = dot_product / (norm_array1 * norm_array2)
        return similarity
    
    # search
    def db_search(self, query_emb, topk=1, threshold=0.9, with_score=False):
        """
            input:
                query_emb, ndarray: the batch embedding of the query contents,
                topk, int: the total number for limiting the search result,
                threshold, float: the threshold for searching.
            output:
                batch_index, list: the retrieved res of the input query embedding.
        # """
        batch_index = []
        scores = []
        if self.index.ntotal == 0:
            for i in range(query_emb.shape[0]):
                batch_index.append([])
            if with_score:
                for i in range(query_emb.shape[0]):
                    scores.append([])
                return batch_index, scores
            return batch_index
        
        D, I = self.index.search(query_emb, topk)  
        for distance_, index_ in zip(D, I):
            index_search_res = []
            score_res = []
            for i,d in enumerate(distance_):
                if d <= threshold:
                    index_search_res.append(index_[i])
                    score_res.append(distance_[i])
            batch_index.append(index_search_res)
            scores.append(score_res)
        if with_score:
            return batch_index, scores
        return batch_index
    
    # search answer
    def db_search_res(self, query_emb, topk=1, threshold=0.9):
        """
            input:
                query_emb, ndarray: the embedding of the query content,
                topk, int: the total number for limiting the search result,
                threshold, float: the threshold for searching,
            output:
                search_answer, str: the searched answer of the content in the db.
                search_content, str: the searched content in the db,
                search_index, int: the searched content's centroid in the db.
        """
        search_res = self.db_search(query_emb, topk, threshold)
        search_content = ''
        search_answer = ''
        index_ = 0
        score = 0
        if search_res:
            for index_ in search_res:
                answer = self.cluster[index_]['centroid_answer']
                centroid_content_idx = self.cluster[index_]['centroid_content_idx']
                search_content = self.cluster[index_]['contents'][ centroid_content_idx]
                # centroid content's embedding 
                centroid_content_emb = self.cluster[index_]['content_index'].reconstruct(centroid_content_idx).reshape(1,-1)
                score = self.cal_sim_score(query_emb, centroid_content_emb)
                answer_t = self.answer_is_invalid_filter(answer)
                if not answer_t:
                    return search_content,  answer, index_, score
                else:
                    answers = self.cluster[index_]['answers']
                    answers = sorted(answers.keys(), key=lambda x: answers[x][0], reverse=True)
                    for answer in answers:
                        if self.answer_is_valid_filter(answer):
                            return search_content,  answer, index_,  score
                    return  search_content,  search_answer, index_, score
        else:
            return search_content, search_answer, index_,  score
    
    
    # update answer offline 
    def update_answer_offline(self):
        """
            update the answer
        """
        for centroid in self.cluster:
            answer_dic = self.cluster[centroid]['answers']
            self.cluster[centroid]['centroid_answer'] = self.vote_answer(self.cluster[centroid]['answers'])
    
    
    def update_content_offline(self):
        """
            update the answer
        """
        for centroid in self.cluster:
            centroid_emb = self.index.index.reconstruct(centroid).reshape(1,-1)
            centroid_content_idx = self.sub_db_search(centroid_emb,  self.cluster[centroid]['content_index'])
            centroid_content_idx = int(centroid_content_idx)
            self.cluster[centroid]['centroid_content_idx'] = centroid_content_idx
        
        
    # sub search
    def sub_db_search(self, query_emb, db_index):
        """
            input:
                query_emb, ndarray: the embedding of the query content,
                db_index, faiss.index: the sub database to save the all content embedding.
            output:
                search_res, int: the retrieved index of the query embedding.
        """
        D, I = db_index.search(query_emb, 1)
        search_index = [i[0] for i in I]
        return search_index[0]

    # update index
    def update_index(self, centroid_id, emb):
        """
            input:
                centroid_id, ndarray(np.int(64)): the centroid index of the emb,
                emb, ndarray: the embedding of the centroid content.
            output:
                None
        """
        self.index.add_with_ids(emb, centroid_id)
        
    def remove_index(self, centroid):
        """
            input:
                centroid_id, ndarray(np.int(64)): the centroid index of the emb,
            output:
                None
        """
        self.index.remove_ids(centroid)
        

    def vote_answer(self, answer_dic):
        """
            input:
                answer_dic, dic: the all answer statistic dictionary.
            output:
                centroid_answer, str: the answer which counts the most.
        """
        try:
            centroid_res = sorted(answer_dic.items(), key=lambda x:(x[1][0]/x[1][1]),reverse=True)[0]
            centroid_answer = centroid_res[0]
        except:
            centroid_answer = ''
        return centroid_answer



    # update
    def db_update(self, update_index, query_emb, query_content, answer=None):
        """
            input:
                update_index, list(int): all index which need to be update,
                query_emb, ndarray: the embedding of the query embedding,
                query_content, str: the query content,
                answer, str: the answer of the query data,
            output:
                None
        """
        # append & update
        for index_ in update_index:
            k = len(self.cluster[index_]['contents'])
            self.cluster[index_]['contents'].append(query_content)
            self.cluster[index_]['content_index'].add(query_emb)
            ori_idx = self.cluster[index_]['centroid_content_idx']
            ori_centroid_emb = self.cluster[index_]['content_index'].reconstruct(ori_idx).reshape(1,-1)
            new_centroid_emb = ((ori_centroid_emb * k) + query_emb )/(k+1)
            new_centroid_emb = new_centroid_emb.astype('float32')
            new_idx = self.sub_db_search(new_centroid_emb,  self.cluster[index_]['content_index'])
            new_idx = int(new_idx)
            answers = self.cluster[index_]['answers']
            flag = True
            # update content
            if new_idx != ori_idx:
                # update centroid idx & centroid representation
                index_new = np.array([index_]).astype(np.int64)
                self.remove_index(index_new)
                self.update_index(index_new, new_centroid_emb)
                self.cluster[index_]['centroid_content_idx'] = new_idx
                flag = False
            # update answer
            self.update_answers_dic(index_, answer, answers, flag)

    def update_answers_dic(self,index_, answers, answer, flag):
        if isinstance(answer,str):
            if not self.answer_is_invalid_filter(answer):
                if answer not in answers:
                    answers[answer] = [0, 0, ''] # score, count
                answers[answer][1] += 1
        elif isinstance(answer,dict):
            for a in answer:
                if not self.answer_is_invalid_filter(a):
                    if a not in answers:
                        answers[a] = [0,0,''] # score, count
                    answers[a][0] += answer[a][0]
                    answers[a][1] += answer[a][1]
                    if answer[a][2]:
                        answers[a][2] = answer[a][2]
        self.cluster[index_]['answers'] = answers
        if not flag:
            self.cluster[index_]['centroid_answer'] = self.vote_answer(answers)
       
    # insert
    def db_insert(self, query_emb, query_content, answer=None):
        """
            input:
                query_emb, ndarray: the embedding of the query embedding,
                query_content, str: the query content,
                answer, str: the answer of the query data,
            output:
                None
        """
        # new centroid
        new_centroid = len(self.cluster)
        new_centroid_ = new_centroid
        new_centroid = np.array([new_centroid]).astype(np.int64)
        # append & update
        self.update_index(new_centroid, query_emb)
        answers = {}
        if isinstance(answer,str):
            if not self.answer_is_invalid_filter(answer):
                answers[answer] = [1,0,''] # score, count 
        elif isinstance(answer, dict):
            for answer_ in answer:
                if not self.answer_is_invalid_filter(answer_):
                    if answer_ not in answers:
                        answers[answer_] = [0,0,'']
                    answers[answer_][0] += 1
                    answers[answer_][1] += 1
        centroid_answer = '' 
        self.cluster[new_centroid_] = {'contents':[query_content],'centroid_content_idx': 0,'centroid_answer': centroid_answer,'answers': answers, 'content_index': faiss.IndexFlatL2(self.dim)}
        self.cluster[new_centroid_]['content_index'].add(query_emb)
        
     
    # remove
    def db_remove(self, query_emb):
        pass
    

    # save db
    def db_save(self, db_dir):
        """
            input:
                db_dir, str: the save directory of the db.
            output:
                None
        """
        index_path = os.path.join(db_dir,'index.faiss')
        faiss.write_index(self.index, index_path)
        db_path = os.path.join(db_dir,'db.bank')
        with open(db_path,'wb') as fw:
            pickle.dump(self.cluster, fw)
        print('save db succeed, save dir: {}'.format(db_dir))

    # load db
    def db_load(self, db_dir):
        """
            input:
                db_dir, str: the save directory of the db.
            output:
                None
        """
        index_path = os.path.join(db_dir,'index.faiss')
        self.index = faiss.read_index(index_path)
        db_path = os.path.join(db_dir,'db.bank')
        with open(db_path,'rb') as fr:
            self.cluster = pickle.load(fr)
            
    def show_db_info(self):
        l = len(self.cluster)
        print('cluster_nums: {}'.format(self.index.ntotal))
        t_max = 0
        t_min = 10000000
        t_cnt = 0
        t_max_cnt = 0
        t_min_cnt = 0
        for c in self.cluster:
            t = len(self.cluster[c]['contents'])
            t_cnt += t
            t_max = max(t_max, t)
            t_min = min(t_min, t)
        for c in self.cluster:
            t = len(self.cluster[c]['contents'])
            if t == t_max:
                t_max_cnt += 1
            if t == t_min:
                t_min_cnt += 1 
        t_cnt_ori = round(t_cnt/l,2)
        try:
            t_cnt = round((t_cnt-t_min_cnt)/(l-t_min_cnt),2)
        except:
            t_cnt = 0  
        print('Max Cluster Nums：{}，Nums: {}'.format(t_max,t_max_cnt))
        print('Min Cluster Nums：{}，Nums: {}'.format(t_min,t_min_cnt))
       


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    query_bank = QueryBank()
    data_path = 'demo_data/t_o.jsonl'
    npy_path = 'demo_data/t.npy'
    # load data
    # query_bank.load_data(data_path, npy_path)
    # cluster
    query_bank._init_db(npy_path, data_path, 1, 10.0)
    # check info
    query_bank.show_db_info()
    # save db
    query_bank.db_save('demo_data/db_save')


   