import os
import re
import random
import nltk
from nltk.tokenize import sent_tokenize
from test_pdf_reader import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertTokenizer, BertModel
import torch

class ChunkSize(object):
    def __init__(self):
        pass

    def gen_static_chunk_res(self, page_text, chunk_size):
        return [page_text[i:i+chunk_size] for i in range(0,len(page_text),chunk_size)]

    def gen_static_chunk_res_by_sliding(self, page_text, window_size, step_size):
        return [page_text[i:i + window_size] for i in range(0, len(page_text) - window_size + 1, step_size)]

    def gen_static_chunk_res_by_overlap(self, page_text, overlap_len):
        total_length = len(page_text)
        return  [page_text[i:i+max_len] for i in range(0,total_length,overlap_len)]
        
    def gen_random_chunk_res(self, page_text, min_chunk_size, max_chunk_size):
        chunks = []
        i = 0
        while i < len(page_text):
            chunk_size = random.randint(min_chunk_size, max_chunk_size)
            chunks.append(page_text[i:i + chunk_size])
            i += chunk_size
        return chunks

    def gen_delimiter_based_chunk_res(self, page_text, delimiter):
        chunks = []
        current_chunk = []
        for item in page_text:
            if item == delimiter:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            else:
                current_chunk.append(item)
        if current_chunk:
            chunks.append(current_chunk)
        chunks = [''.join(chunks[i]) for i in range(len(chunks))]
        return chunks

    def gen_condition_based_chunk_res(self, page_text, condition_func):
        chunks = []
        current_chunk = []
        for item in page_text:
            if condition_func(item):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            current_chunk.append(item)
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
 
    def gen_sentence_based_chunk_res(self, page_text):
        sentences = nltk.sent_tokenize(page_text)
        return sentences

    def gen_paragraph_based_chunk_res(self, page_text):
        paragraphs = page_text.split('\n\n')
        return paragraphs

    def gen_topic_based_chunk_res(self, page_text, num_topics=5):
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(page_text.split('\n\n'))
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(X)
        topic_chunks = []
        for topic_idx, topic in enumerate(lda.components_):
            topic_chunks.append(" ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]))
        return topic_chunks

    def gen_embedding_based_chunk_res(self, page_text, chunk_size=512):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        inputs = tokenizer(page_text, return_tensors='pt', max_length=chunk_size, truncation=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        chunks = []
        for i in range(0, embeddings.size(1), chunk_size):
            chunks.append(embeddings[:, i:i + chunk_size, :])
        return chunks
    


if __name__=='__main__':
    pdf_reader = PDFReader()
    pdf_path = os.path.join(current_path, '../demo_data/t.pdf')
    s = pdf_reader.extract_pdf_page_text_pymupdf(pdf_path)
    s = ''.join(s)

    chunk_size_generator = ChunkSize()
    print("Fixed Size Chunks:")
    max_len = 256
    print(chunk_size_generator.gen_static_chunk_res(s,max_len))

    print("\nSliding Window Chunks:")
    step_size =100
    print(chunk_size_generator.gen_static_chunk_res_by_sliding(s,  max_len, step_size))

    print("\nSliding Window Chunks with Overlap:")
    overlap_len = 100
    print(chunk_size_generator.gen_static_chunk_res_by_overlap(s, overlap_len))

    print("\nRandom Size Chunks:")
    min_chunk_size = 100
    print(chunk_size_generator.gen_random_chunk_res(s, min_chunk_size, max_len))

    print("\nDelimiter Based Chunks:")
    delimiter = '。'
    print(chunk_size_generator.gen_delimiter_based_chunk_res(s,delimiter))

    # print("Sentence Based Chunks:")
    # print(chunk_size_generator.gen_sentence_based_chunk_res(s))

    print("\nParagraph Based Chunks:")
    print(chunk_size_generator.gen_paragraph_based_chunk_res(s))

    print("\nTopic Based Chunks:")
    print(chunk_size_generator.gen_topic_based_chunk_res(s))

    # print("\nEmbedding Based Chunks:")
    # print(chunk_size_generator.gen_embedding_based_chunk_res(s))

 
