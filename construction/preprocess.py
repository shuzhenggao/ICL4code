import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import jsonlines
import random
import numpy as np
from gensim import corpora
from gensim.summarization import bm25
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
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





def bm25_preprocess(train, test, output_path, number):
    code = [' '.join(obj['code_tokens']) for obj in train]
    bm25_model = bm25.BM25(code)
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    
    processed = []
    for obj in tqdm(test, total=len(test)):
        query = obj['code_tokens']
        score = bm25_model.get_scores(query,average_idf)
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:number]
        code_candidates_tokens = []
        for i in range(len(rtn)):
            code_candidates_tokens.append({'code_tokens': train[rtn[i][0]]['code_tokens'], 'docstring_tokens': train[rtn[i][0]]['docstring_tokens'], 'score': rtn[i][1], 'idx':i+1})            
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
        
    with jsonlines.open(os.path.join(output_path, 'test_bm25_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)





def oracle_preprocess(train, test, output_path, number):
    code = [' '.join(obj['docstring_tokens']) for obj in train]
    bm25_model = bm25.BM25(code)
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    
    processed = []
    for obj in tqdm(test, total=len(test)):
        query = obj['docstring_tokens']
        score = bm25_model.get_scores(query,average_idf)
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:number]
        code_candidates_tokens = []
        for i in range(len(rtn)):
            code_candidates_tokens.append({'code_tokens': train[rtn[i][0]]['code_tokens'], 'docstring_tokens': train[rtn[i][0]]['docstring_tokens'], 'score': rtn[i][1], 'idx':i+1})  
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
        
    with jsonlines.open(os.path.join(output_path, 'test_oracle_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)





def sbert_preprocess(train, test, model_path, output_path, number):
    code = [' '.join(obj['code_tokens']) for obj in train]
    other = [' '.join(obj['docstring_tokens']) for obj in train]
    
    model = SentenceTransformer(model_path)
    code_emb = model.encode(code, convert_to_tensor=True)
    processed = []
    for obj in tqdm(test):
        query = ' '.join(obj['code_tokens'])
        query_emb = model.encode(query , convert_to_tensor=True)
        hits = util.semantic_search(query_emb, code_emb, top_k=number)[0]
        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': code[hits[i]['corpus_id']].split(),'docstring_tokens': other[hits[i]['corpus_id']].split(), 'score': hits[i]['score'], 'idx':i+1})
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
    
    with jsonlines.open(os.path.join(output_path, 'test_sbert_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)




def unixcoder_cocosoda_preprocess(train, test, model_path, output_path, number):
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
    
    test_emb = []
    model.eval()
    for obj in tqdm(test):
        with torch.no_grad():
            code_tokens=tokenizer.tokenize(' '.join(obj['code_tokens']))[:256-4]
            tokens=[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings=model(code_inputs=torch.tensor([tokens_ids]).to(device))
            test_emb.append(context_embeddings.cpu().numpy())
    test_emb = np.concatenate(test_emb,0)

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score':cos_sim[i], 'corpus_id':i})
        
        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'], 'score': float(hits[i]['score']), 'idx':i+1})
        obj = test[idx]
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
    model_name = 'unixcoder' if 'unixcoder' in model_path else 'cocosoda'
    with jsonlines.open(os.path.join(output_path, 'test_'+str(model_name)+'_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)





train = []
with jsonlines.open('/data/szgao/resource/dataset/CodeSearchNet/java/train.jsonl') as f:
    for i in f:
        train.append(i)
test = []
with jsonlines.open('/data/szgao/resource/dataset/CodeSearchNet/java/test.jsonl') as f:
    for i in f:
        test.append(i)

oracle_preprocess(train, test, './preprocess', 64)
bm25_preprocess(train, test, './preprocess', 64)
sbert_preprocess(train, test, '/data/szgao/resource/pretrain/flax-sentence-embeddings/st-codesearch-distilroberta-base', './preprocess', 64) #https://huggingface.co/flax-sentence-embeddings/st-codesearch-distilroberta-base
unixcoder_cocosoda_preprocess(train, test, '/data/szgao/resource/pretrain/microsoft/unixcoder-base', './preprocess', 64) #https://huggingface.co/microsoft/unixcoder-base
unixcoder_cocosoda_preprocess(train, test, '/data/szgao/resource/pretrain/microsoft/cocosoda', './preprocess', 64) #https://huggingface.co/DeepSoftwareAnalytics/CoCoSoDa

