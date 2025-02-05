import docx
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pdb


model_id = "meta-llama/Llama-3.2-3B-Instruct"
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, config=config).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_id, config=config, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
chat_template = open("llama-3-instruct.jinja").read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer.chat_template = chat_template

system_prompt = """
Extract all the names (if any) of persons and organizations from the following text.
"""

file_path = "AML Policy_Vault AM_v1_comments MvK.docx"
doc = docx.Document(file_path)
for para in doc.paragraphs:
    if para.text == "":
        continue
    messages = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': para.text,
        }
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True).to('cuda')
    generated_ids = model.generate(encodeds, max_new_tokens=100, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    result = decoded[0].split('assistant')[1].strip()
    print("=========================================")
    print(result)
    print("=========================================")
    pdb.set_trace()










