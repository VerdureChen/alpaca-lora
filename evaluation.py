'''
参考了RankGPT的评估代码
'''

THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever.flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',

}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',


}


from pyserini.search import LuceneSearcher, get_topics, get_qrels
from tqdm import tqdm
import tempfile
from rank_llama import run_retriever, sliding_windows, write_eval_file

import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass



def evaluate(
        model,
        tokenizer,
        prompter,
        generate_config: GenerationConfig,
        data_list: list = ['dl19', 'dl20', 'covid', 'arguana', 'touche', 'news', 'scifact', 'fiqa', 'scidocs', 'nfc', 'quora', 'dbpedia', 'fever', 'robust04', 'signal'],
        lora_weights: str = "tloen/alpaca-lora-7b",
        rank_end: int = 100,
):
    for data in data_list:
        print('#' * 20)
        print(f'Evaluation on {data}')
        print('#' * 20)

        # Retrieve passages using pyserini BM25.
        searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
        topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(THE_TOPICS[data])
        rank_results = run_retriever(topics, searcher, qrels, k=100)

        # Evaluate nDCG@10
        from trec_eval import EvalFunction
        temp_file_bm25 = f'result/{lora_weights.split("/")[-1]}/{data}_{rank_end}_BM25.txt'
        write_eval_file(rank_results, temp_file_bm25)
        EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], temp_file_bm25])

        # Run sliding window permutation generation
        new_results = []
        for i, item in enumerate(tqdm(rank_results)):
            # if i <200:
            #     continue

            new_item = sliding_windows(item, rank_start=0, rank_end=rank_end, window_size=10, step=5,
                                       model=model, tokenizer=tokenizer, prompter=prompter, generate_config=generate_config,
                                       device=device)
            new_results.append(new_item)
            # break


        # temp_file = tempfile.NamedTemporaryFile(delete=False).name
        # the path of the temp_file is in the /result/{lora_weights}.split('/')directory on the root of the project
        if not os.path.exists(f'result/{lora_weights.split("/")[-1]}'):
            os.makedirs(f'result/{lora_weights.split("/")[-1]}')

        temp_file = f'result/{lora_weights.split("/")[-1]}/{data}_{rank_end}.txt'
        write_eval_file(new_results, temp_file)
        EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], temp_file])


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
        data_list: str='dl19,dl20,covid,arguana,touche,news,scifact,fiqa,scidocs,nfc,quora,dbpedia,fever,robust04,signal',
):
    if type(data_list) == str:
        data_list = data_list.split(',')
    else:
        data_list = list(data_list)
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print('no lora')
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def eva_config(
        model,
        tokenizer,
        prompter,
        data_list: list = ['dl19', 'dl20', 'covid', 'arguana', 'touche', 'news', 'scifact', 'fiqa', 'scidocs', 'nfc', 'quora', 'dbpedia', 'fever', 'robust04', 'signal'],
        lora_weights=lora_weights,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
    ):
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        evaluate(model, tokenizer, prompter, generation_config, data_list, lora_weights=lora_weights)

    eva_config(model, tokenizer, prompter, data_list, lora_weights=lora_weights)


if __name__ == '__main__':
    fire.Fire(main)