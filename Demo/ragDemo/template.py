# solve problem
def build_template():
    prompt_template = r'''<|im_start|>user\n你是一个准确和可靠的人工智能助手，能够借助外部文档回答用户问题，请注意外部文档可能存在噪声事实性错误。
                      如果文档中的信息包含了正确答案，请先生成“文档信息正确，基于提供的文档可以回答该问题。”，并给出你的答案。
                      如果文档中的信息不包含答案，请先生成“文档信息不足，基于提供的文档无法回答该问题。”，并给出你的参考答案。
                      如果部分文档中存在与事实不一致的错误，请先生成“文档信息错误，提供的内容中存在事实性错误。”，并生成正确答案。
                      下面给定你相关外部文档，根据文档来回答用户问题。
                        外部文档：%s
                        用户问题：%s
                        <|im_end|>\n<|im_start|>assistant'''
    return prompt_template

# rewrite question
def build_rewrite_template():
    prompt_template = r'''<|im_start|>user\n你是一个准确和可靠的人工智能助手，能够抓住问题中的关键内容，并据此对问题进行改写，请注意不要改变原始问题的意思。
                        如果原始问题中包含了冗余信息，请你对原始问题进行简化，去掉冗余信息，但不能改变原始问题的含义。
                        如果原始问题中不包含多余信息，请你对原始问题进行改写，需要以更加简洁明了并且有助于你进行理解和回答的方式表示原始问题。
                        下面给定你一个用户问题，请根据问题内容进行改写。注意不要有冗余信息，例如不想关的字符。%s
                        改写问题：%s
                        <|im_end|>\n<|im_start|>assistant'''
    return prompt_template

# rewrite answer
def build_answer_rewrite_template():
    prompt_template = r'''<|im_start|>user\n你是一个准确和可靠的人工智能助手，能够抓住答案中的关键内容，并据此对答案进行改写，请注意答案中可能存在噪音或事实性错误。
                        如果原始答案中，请你对原始问题进行简化，去掉冗余信息，但不能改变原始问题的含义。
                        如果原始问题中不包含多余信息，请你对原始问题进行改写，需要以更加简洁明了并且有助于你进行理解和回答的方式表达原始问题。
                        下面给定你一个用户问题，请根据问题内容进行改写。
                        原始问题：%s
                        原始答案：%s
                        <|im_end|>\n<|im_start|>assistant'''
    return prompt_template

# rewrite retrieval result
def build_retrieval_rewrite_template():
    prompt_template = r'''<|im_start|>user\n你是一个准确和可靠的人工智能助手，能够抓住答案中的关键内容，并据此对答案进行改写，请注意答案中可能存在噪音或事实性错误。
                        如果原始答案中，请你对原始问题进行简化，去掉冗余信息，但不能改变原始问题的含义。
                        如果原始问题中不包含多余信息，请你对原始问题进行改写，需要以更加简洁明了并且有助于你进行理解和回答的方式表达原始问题。
                        下面给定你一个用户问题，请根据问题内容进行改写。
                        原始问题：%s
                        原始答案：%s
                        改写问题：%s
                        <|im_end|>\n<|im_start|>assistant'''
    return prompt_template
