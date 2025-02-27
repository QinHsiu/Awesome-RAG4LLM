from huggingface_hub import snapshot_download



from transformers import BertTokenizer, BertModel

def download_kokoro_model():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='')
    model = BertModel.from_pretrained(model_name, cache_dir='')
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = download_kokoro_model()
    print("Model and tokenizer downloaded successfully.")