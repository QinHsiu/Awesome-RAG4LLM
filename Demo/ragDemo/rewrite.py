from template import build_rewrite_template, build_answer_rewrite_template, build_retrieval_rewrite_template
from llm import *
current_path = os.path.dirname(os.path.abspath(__file__))

class Rewrite(object):
    def __init__(self, llm_model_name_or_path, **kwargs):
        self.rewrite_mode = kwargs.get('write_mode','query_rewrite')
        self.llm = LLMPredictor(llm_model_name_or_path, template_mode=self.rewrite_mode, **kwargs)
        self.template = self.llm.build_template4infer(self.rewrite_mode)


    def rewrite(self, inputs, input_file, output_file, output_key):
        self.llm.model_infer(inputs, input_file, output_file, output_key)
      


if __name__=='__main__':
    # set cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    llm_model_name_or_path = os.path.join(current_path, 'llm/Qwen2.5-7B-Instruct')
    input_file = os.path.join(current_path, 'demo_data/t.jsonl')
    output_file = os.path.join(current_path, 'demo_data/t_r.jsonl')
    device = 'cuda'
    data = load_json_data(input_file)[0]
    inputs=(data['retrieve_doc'],data['question'])
    rewritor = Rewrite(llm_model_name_or_path, device=device)
    rewritor.rewrite(inputs, input_file, output_file,'rewrite_predict')


        


        
