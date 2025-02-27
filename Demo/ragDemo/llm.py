import json
import os
from vllm import LLM, SamplingParams
from template import *
from utils import *
current_path = os.path.dirname(os.path.abspath(__file__))


class LLMPredictor(object):
    def __init__(self, llm_model_name_or_path, device='cuda', **kwargs):
        self.kwargs = kwargs 
        self.max_token = self.kwargs.get('max_token', 4096)
        self.temperature = self.kwargs.get('temperature', 1.0)
        self.topk = self.kwargs.get('topk', 2)
        self.top_p = self.kwargs.get('top_p', 0.95)
        # self.use_beam_search = self.kwargs.get('use_beam_search', False)
        self.best_of = self.kwargs.get('best_of', 1)
        self.num_gpus = self.kwargs.get('num_gpus', 1)
        self.template_mode = self.kwargs.get('template_mode','infer')
        self.infer_template = self.build_template4infer(self.template_mode)
        self.device = device
        self.sample_params = SamplingParams(
            max_tokens=self.max_token,
            temperature=self.temperature,
            top_k=self.topk,
            top_p=self.top_p,
            best_of=self.best_of
        )
        self.model_name_or_path = llm_model_name_or_path
        self.llm = LLM(model=llm_model_name_or_path, tensor_parallel_size=self.num_gpus, dtype='float16')

    def build_template4infer(self,template_mode):
        if template_mode == 'infer':
            return build_template()
        elif template_mode == 'query_rewrite':
            return build_rewrite_template()
        elif template_mode == 'answer_rewrite':
            return build_answer_rewrite_template()
        elif template_mode == 'retrieval_rewrite':
            return build_retrieval_rewrite_template()
        else:
            raise ValueError('Error mode, please use [infer,query_rewrite,answer_rewrite,retrieval_rewrite]')
        _
    # single question
    def build_prompt(self, doc, question):
        if doc is None:
            return self.infer_template%('', question)
        else:
            doc = doc.replace('\n', '')
            return self.infer_template%(doc, question)
    
    # muilple questions
    def build_prompts(self, docs, questions):
        if docs is None:
            return [self.infer_template%('', question) for question in questions]
        else:
            docs = docs.replace('\n', '')
            return [self.infer_template%(doc, question) for doc,question in zip(docs,questions)]

    # post process
    def post_process_outputs(self, input_file, output_file, outputs, output_key='model_predict'):
        prompt_list = []
        response_list = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            prompt_list.append(prompt)
            response_list.append(generated_text)
        
        with open(input_file, 'r') as fr, open(output_file, 'w') as fw:
            for idx, line in enumerate(fr.readlines()):
                line = json.loads(line.strip())
                line[output_key] = response_list[idx]
                fw.write(json.dumps(line, ensure_ascii=False) + "\n")

    # model infer
    def model_infer(self, inputs, input_file, output_file,output_key='model_predict'):
        docs, questions = inputs
        if isinstance(questions,list):
            prompts = self.build_prompts(docs, questions)
        else:
            prompt = self.build_prompt(docs, questions)
            prompts = [prompt]
        outputs = self.llm.generate(prompts=prompts, sampling_params=self.sample_params)
        self.post_process_outputs(input_file, output_file, outputs, output_key)
        
if __name__=='__main__':
    # set cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    llm_model_name_or_path = os.path.join(current_path, 'llm/Qwen2.5-7B-Instruct')
    input_file = os.path.join(current_path, 'demo_data/t.jsonl')
    output_file = os.path.join(current_path, 'demo_data/t_o.jsonl')
    device = 'cuda'
    data = load_json_data(input_file)[0]
    inputs=(data['retrieve_doc'],data['question'])
    llm_predictor = LLMPredictor(llm_model_name_or_path, device=device)
    llm_predictor.model_infer(inputs, input_file, output_file,'model_predict')


