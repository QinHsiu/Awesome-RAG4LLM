import json

def load_json_data(dp):
    with open(dp,'r+') as fr:
        data = fr.readlines()
    data = [json.loads(d.strip()) for d in data]
    return data


def pre_process(page_text):
    new_text = [text.strip() for text in page_text.split('\n')]
    new_text = '\n'.join(new_text)
    return new_text

