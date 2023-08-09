import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import jsonlines
import random
import numpy as np
from gensim import corpora
from gensim.summarization import bm25
from transformers import AutoTokenizer, AutoModel, RobertaModel
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


def random_selection(train, test, max_length, number, insturction, input_insturction, output_insturction, output_path):
    random.shuffle(train)
    selected = random.sample(list(range(0,len(train)-1)), number) #to revise
    prompt = ''
    instance_length = max_length//(number+1)
    format_length = len((insturction+input_insturction+output_insturction).split())
    for idx in selected:
        prompt += insturction
        prompt += input_insturction
        prompt += ' '.join(train[idx]['code_tokens'][:instance_length - format_length -len(train[idx]['docstring_tokens'])]).strip()+'\n'
        prompt += output_insturction
        prompt += ' '.join(train[idx]['docstring_tokens']).strip()
        prompt += '\n\n'
    prompts = []
    for obj in test:
        tmp_prompt = prompt + insturction + input_insturction
        tmp_prompt += ' '.join(obj['code_tokens'][:instance_length - format_length]).strip()+'\n'
        tmp_prompt += output_insturction
        prompts.append({'prompt':tmp_prompt, 'label':' '.join(obj['docstring_tokens'])})
    with jsonlines.open(os.path.join(output_path, 'summary_random_random_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(prompts)



def randomkmeans_selection(train, test, max_length, number, insturction, input_insturction, output_insturction, output_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model)
    model.to(device)

    code_emb = []
    model.eval()
    for obj in tqdm(train):
        with torch.no_grad():
            code_tokens=tokenizer.tokenize(' '.join(obj['code_tokens']))[:256-4]
            tokens=[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings=model(code_inputs=torch.tensor([tokens_ids]).to(device))
            code_emb.append(context_embeddings.cpu().numpy())
    code_emb = np.concatenate(code_emb,0)
    
    number = 4
    cluster_number = number
    clf = KMeans(n_clusters=cluster_number, init='k-means++') 
    code_label = clf.fit_predict(code_emb)
    class_pos = {}
    selected = []
    for idx in range(len(code_label)):
        i = code_label[idx]
        if i not in class_pos:
            class_pos[i] = [idx]
        else:
            class_pos[i].append(idx)
    for i in class_pos:
        pos = random.randint(0,len(class_pos[i])-1) 
        selected.append(class_pos[i][pos])

    prompt = ''
    instance_length = max_length//(number+1)
    format_length = len((insturction+input_insturction+output_insturction).split())
    for idx in selected:
        prompt += insturction
        prompt += input_insturction
        prompt += ' '.join(train[idx]['code_tokens'][:instance_length - format_length -len(train[idx]['docstring_tokens'])]).strip()+'\n'
        prompt += output_insturction
        prompt += ' '.join(train[idx]['docstring_tokens']).strip()
        prompt += '\n\n'
    prompts = []
    for obj in test:
        tmp_prompt = prompt + insturction + input_insturction
        tmp_prompt += ' '.join(obj['code_tokens'][:instance_length - format_length]).strip()+'\n'
        tmp_prompt += output_insturction
        prompts.append({'prompt':tmp_prompt, 'label':' '.join(obj['docstring_tokens'])})
    with jsonlines.open(os.path.join(output_path, 'summary_randomkmeans_random_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(prompts)



def instance_selection(method, max_length, number, insturction, input_insturction, output_insturction, preprocess_file, output_path):
    test = []
    with jsonlines.open(preprocess_file) as f:
        for i in f:
            test.append(i)
    prompts = []
    instance_length = max_length//(number+1)
    format_length = len((insturction+input_insturction+output_insturction).split())
    for idx,obj in enumerate(test):
        topk = []
        for sample in obj['code_candidates_tokens']:
            if sample['idx']<=number:
                topk.append(sample)

        prompt = ''
        random.shuffle(topk)
        for sample in topk:
            prompt += insturction
            prompt += input_insturction
            prompt += ' '.join(sample['code_tokens'][:instance_length - format_length -len(sample['docstring_tokens'])]).strip()+'\n'
            prompt += output_insturction
            prompt += ' '.join(sample['docstring_tokens']).strip()
            prompt += '\n\n'
        tmp_prompt = prompt + insturction + input_insturction
        tmp_prompt += ' '.join(obj['code_tokens'][:instance_length - format_length]).strip()+'\n'
        tmp_prompt += output_insturction
        prompts.append({'prompt':tmp_prompt, 'label':' '.join(obj['docstring_tokens'])})
            
    with jsonlines.open(os.path.join(output_path, 'summary_'+str(method)+'_random_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(prompts)



def order(test, method, max_length, number, insturction, input_insturction, output_insturction, output_path):
    data = []
    with jsonlines.open(os.path.join(output_path, 'summary_'+str(method)+'_random_'+str(number)+'.jsonl')) as f:
        for i in f:
            data.append(i)
    code = []
    train_dict = {}
    for i in data[0]['prompt'].split('\n\n')[:number]:
        inputs, outputs = i.split('[output]')[0], i.split('[output]')[1].strip()
        inputs = inputs.split('[input]')[1].strip()
        code.append(inputs)
        train_dict[inputs] = outputs
    bm25_model = bm25.BM25(code)
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())

    prompts1 = []
    prompts2 = []
    instance_length = max_length//(number+1)
    format_length = len((insturction+input_insturction+output_insturction).split())
    for obj in test:
        query = obj['code_tokens']
        score = bm25_model.get_scores(query,average_idf)
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:number]
        sorted_topk=[]
        for i in range(len(rtn)):
            sorted_topk.append({'code_tokens':code[rtn[i][0]].split()})
            
        prompt1 = ''
        for sample in sorted_topk:
            string = ' '.join(' '.join(sample['code_tokens']).split())
            prompt1 += insturction
            prompt1 += input_insturction
            prompt1 += ' '.join(sample['code_tokens'][:instance_length - format_length -len(train_dict[string].split())]).strip()+'\n'
            prompt1 += output_insturction
            prompt1 += train_dict[string].strip()
            prompt1 += '\n\n'
        tmp_prompt1 = prompt1 + insturction + input_insturction
        tmp_prompt1 += ' '.join(obj['code_tokens'][:instance_length - format_length]).strip()+'\n'
        tmp_prompt1 += output_insturction
        prompts1.append({'prompt':tmp_prompt1, 'label':' '.join(obj['docstring_tokens'])})
        
        prompt2 = ''
        for sample in sorted_topk[::-1]:
            string = ' '.join(' '.join(sample['code_tokens']).split())
            prompt2 += insturction
            prompt2 += input_insturction
            prompt2 += ' '.join(sample['code_tokens'][:instance_length - format_length -len(train_dict[string].split())]).strip()+'\n'
            prompt2 += output_insturction 
            prompt2 += train_dict[string].strip()
            prompt2 += '\n\n'
        tmp_prompt2 = prompt2 + insturction + input_insturction
        tmp_prompt2 += ' '.join(obj['code_tokens'][:instance_length - format_length]).strip()+'\n'
        tmp_prompt2 += output_insturction
        prompts2.append({'prompt':tmp_prompt2, 'label':' '.join(obj['docstring_tokens'])})

    with jsonlines.open(os.path.join(output_path, 'summary_'+str(method)+'_order1_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(prompts1)
    with jsonlines.open(os.path.join(output_path, 'summary_'+str(method)+'_order2_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(prompts2)



train = []
with jsonlines.open('/data/dataset/CodeSearchNet/java/train.jsonl') as f:
    for i in f:
        train.append(i)
test = []
with jsonlines.open('/data/dataset/CodeSearchNet/java/test.jsonl') as f:
    for i in f:
        test.append(i)
number = 4
max_length = 6000
insturction = 'Generate comment (summarization) for this code  \n'
input_insturction = ' [input] '
output_insturction = ' [output] '


#random_selection(train, test, max_length, number, insturction, input_insturction, output_insturction, './data')
#randomkmeans_selection(train, test, max_length, number, insturction, input_insturction, output_insturction, './data', '/data/pretrain/microsoft/unixcoder-base')
#instance_selection('sbert', max_length, number, insturction, input_insturction, output_insturction, './preprocess/test_sbert_64.jsonl', './data')
#instance_selection('unixcoder', max_length, number, insturction, input_insturction, output_insturction, './preprocess/test_unixcoder_64.jsonl', './data')
#instance_selection('cocosoda', max_length, number, insturction, input_insturction, output_insturction, './preprocess/test_cocosoda_64.jsonl', './data')
#instance_selection('bm25', max_length, number, insturction, input_insturction, output_insturction, './preprocess/test_bm25_64.jsonl', './data')
#order(test, 'bm25', max_length, number, insturction, input_insturction, output_insturction, './data')
#order(test, 'random', max_length, number, insturction, input_insturction, output_insturction, './data')
#order(test, 'randomkmeans', max_length, number, insturction, input_insturction, output_insturction, './data')