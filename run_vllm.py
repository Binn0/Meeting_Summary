import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

num_gpus = 8

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data1/rbwei/models/Qwen2.5-72B-Instruct")

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=8192)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model='/data1/rbwei/models/Qwen2.5-72B-Instruct', 
        #   tokenizer_mode="auto", 
          dtype=torch.bfloat16,
          tensor_parallel_size=8)

# Move the model to GPUs
llm = torch.nn.DataParallel(llm, device_ids=list(range(num_gpus)))

audio2text = "this is a text"
prompt = f"""
【会议精要】整理生成高质量会议纪要，保证内容完整、准确且精炼 

你是一个专业的CEO秘书，专注于整理和生成高质量的会议纪要，确保会议目标和行动计划清晰明确。
要保证会议内容被全面地记录、准确地表述。准确记录会议的各个方面，包括议题、讨论、决定和行动计划
保证语言通畅，易于理解，使每个参会人员都能明确理解会议内容框架和结论
简洁专业的语言：信息要点明确，不做多余的解释；使用专业术语和格式
用户提供给你的文本是一段语音转录成的文字，然后需要你帮忙把转录出来的文本整理成没有口语、没有错别字、逻辑清晰、、内容明确的会议纪要

## 工作流程:
- 输入: 用户提供给你的一段语音转录成文字的文本
- 整理: 遵循以下框架来整理用户提供的会议信息，每个步骤后都会进行数据校验确保信息准确性
    - 会议主题：会议的标题和目的。
    - 会议日期和时间：会议的具体日期和时间。
    - 参会人员：列出参加会议的所有人。
    - 会议记录者：注明记录这些内容的人。
    - 会议议程：列出会议的所有主题和讨论点。
    - 主要讨论：详述每个议题的讨论内容，主要包括提出的问题、提议、观点等。
    - 决定和行动计划：列出会议的所有决定，以及计划中要采取的行动，以及负责人和计划完成日期。
    - 会议总结: 对本次会议的全部内容进行一个总结
- 输出: 输出整理后的结构清晰, 描述完整的会议纪要

## 注意:
- 整理会议纪要过程中, 需严格遵守信息准确性, 不对用户提供的信息做扩写
- 仅做信息整理, 将一些明显的病句做微调
- 用户提供的文本是一行一行的格式，对应发言人的一句话
- 提供的文本前面的数字代表某一个发言人，不同的数字是不同的发言人
- 将一些明显的错别字改正，将一些语气词删除
- 对于涉及到的某个专业领域的词汇尽量表达准确
- 会议纪要：一份详细记录会议讨论、决定和行动计划的文档。
- 最后只需要将得到的会议纪要输出就行，不用输出任何多余的其他内容
## 用户提供的语音转录从文字的文本:
{audio2text}
"""

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


# generate outputs
outputs = llm.module.generate([text], sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print("respones: ")
    print(generated_text)
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
