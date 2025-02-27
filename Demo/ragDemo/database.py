from langchain.vectorstores import FAISS
from langchain_chroma import Chroma
from tqdm import tqdm
from reader import Reader
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import os

def test_speed(test_dbs,docs,embedding):
    l=len(docs)
    for idx,db in tqdm(enumerate(test_dbs),total=len(test_dbs)):
        if db=="faiss":
            persist_directory="faiss_db_{}".format(l)
            b=time()
            # create database & save database
            t_db=FAISS.from_documents(documents=docs, embedding=embedding)
            t_db.save_local(persist_directory)
            e=time() 
        else:
            persist_directory="chroma_db_{}".format(l)
            b=time()
            # create database & save database
            # ValueError: Expected metadata value to be a str, int, float or bool, got ['未知'] which is a list
            t_db=Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
            e=time()
        print("data nums {0}, use db {1} to create database costs {2} seconds".format(l,db,e-b))
        query_text="已知(1)酶、(2)抗体、(3)激素、(4)糖原、(5)脂肪、(6)核酸都是人体内有重要作用的物质。下列说法正确的 是 \n(A)(1)(2)(3)都是由氨基酸通过肽键连接而成的\n(B)(3)(4)(5)都是生物大分子, 都以碳链为骨架\n(C)(1)(2)(6)都是由含氮的单体连接成的多聚体\n(D)(4)(5)(6)都是人体细胞内的主要能源物质"
        res=[t for t in t_db.similarity_search_with_score(query_text,k=10)]
        print(res)
        e1=time()
        print("search a query costs {} seconds".format(e1-e))
        # del t_db

if __name__=="__main__":
    # data nums 20000, use db faiss to create database costs 41.63766813278198 seconds
    # search a query costs 26.197269201278687 seconds
    
    # data nums 20000, use db chroma to create database costs 75.09635853767395 seconds
    # search a query costs 29.165228128433228 seconds
    
    # data nums 200000, use db faiss to create database costs 155.8989384174347 seconds
    # search a query costs 31.987764596939087 seconds
    
    # data nums 200000 use db chroma to create database costs 643.0778284072876 seconds
    # search a query costs 28.064088582992554 seconds
    
    # data nums 2000000, use db faiss to create database costs 1384.0919799804688 seconds
    # search a query costs 40.22010946273804 seconds
    
    # data nums 2000000, use db chroma to create database costs 16649.01 seconds
    # search a query costs 17.03 seconds
    
    test_dbs=['faiss','chroma'][1:]
    data_path="demo_data/t_o.jsonl"
    emb_model_name_or_path="llm/bge-large-zh"

    reader = Reader(data_path)
    corpus = reader.corpus
    # print(corpus[0])
    # chroma
    # corpus=[Document(page_content=t,metadata={"query":t}) for t in corpus]
    # faiss
    corpus=[Document(page_content=t['question'],metadata={"query":t['question'],'answer':t['model_predict']}) for t in corpus]
    embedding=HuggingFaceEmbeddings(model_name=emb_model_name_or_path,model_kwargs = {'device': 'cuda'})
    test_speed(test_dbs,corpus,embedding)
    
