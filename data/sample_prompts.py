#transformers==4.10.2
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch

T5_PATH = 't5-base' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# Input text template
text = '<extra_id_0> the list of words: noun1, noun2, noun3. <extra_id_1> the list again: noun1, noun2, noun3. '

encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

# Generaing 20 sequences with maximum length set to 5
outputs = t5_mlm.generate(input_ids=input_ids,
                          num_beams=200, num_return_sequences=20,
                          max_length=10)

span_0_len, span_1_len = 12, 12
span_0_start = text.index('<extra_id_0>')
span_1_start = text.index('<extra_id_1>')
_result_prefix = text[:span_0_start]
_result_infix = text[(span_0_start + span_0_len):span_1_start]  # 12 is the length of <extra_id_0>
_result_suffix = text[(span_1_start+span_1_len):]

def _filter(output, span1='<extra_id_0>', span2='<extra_id_1>', end_token='<extra_id_2>'):
    
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    
    span1_start = _txt.index(span1)
    span2_start = _txt.index(span2)
    span1_len = len(span1)
    span2_len = len(span2)

    if end_token in _txt:
        end_token_start = _txt.index(end_token)
        return _result_prefix + _txt[(span1_start+span1_len):span2_start] + _result_infix + _txt[(span2_start+span2_len):end_token_start] + _result_suffix
    
    else:
        return _result_prefix + _txt + _result_suffix

results = list(map(_filter, outputs))
[e for e in results]