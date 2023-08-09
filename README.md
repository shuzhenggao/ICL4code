This is the replication package for our ASE 2023 paper "What Makes Good In-context Demonstrations for Code Intelligence Tasks with LLMs?". This repo contains the code and full experimental results. We study three aspects of demonstration construction including the selection of demonstration examples, the order of demonstration examples, and the number of demonstration examples.

## Setup
You need to first have some OpenAI API keys and add them in api_keys list in ``{codex,gpt3.5,chatgpt}.py``.

- Package requirement of querying LLM: 

```
pip install tqdm
pip install openai
pip install jsonlines
```

- Package requirement for different prompt construction methods: 

```
pip install numpy
pip install torch
pip install heapq
pip install sklearn
pip install gensim==3.4.0
pip install transformers
pip install sentence_transformers
```

## Prompt construction    
You can first run ``construction/preprocess.py`` to first complete the retrieval process. Then you can run ``construction/construction.py`` to get the prompt files. You can choose different selection methods, ordering methods, demonstration numbers or insturctions in ``construction/construction.py``.  
  
The code is based on code summarization. If you want to adopt it to other tasks, you can first preprocess your data into jsonlines format and change the directory and key name in these two files.


## Experiments
Run ``{codex,gpt3.5,chatgpt}.py`` to get predictions for your prompts stored in files with jsonlines format. For example:
 
```
python chatgpt.py prompts/summary.jsonl
```
The results will be saved in ``prompts/summary_result_.jsonl``. Note that CodeX is not available now since OpenAI has stopped its service.


## Prompt examples 
We show some examples of prompt files in the prompts folder. ``prompts/bug.jsonl`` presents an example of the task-level demonstration and the other two files are instance-level demonstrations.

