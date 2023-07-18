import os
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, Sequence
import transformers
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import datasets
from collections import defaultdict


PROMPT_DICT = {
    "prompt_input": (
        "Determine if the given passage is relevant to the query based on the given instruction. \n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Query:\n{query}\n\n"
        "### Passage:\n{passage}\n\n"
        "### Relevance:\n"
    ),
    "prompt_no_input": (
        "Determine if the given passage is relevant to the query. \n\n"
        "### Query:\n{query}\n\n"
        "### Passage:\n{passage}\n\n"
        "### Relevance:\n"
    ),
}

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm(strings)
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    labels = targets
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        if not os.path.exists(data_path + "_tokenized.pkl"):
            logging.warning("Loading data...")
            # list_data_dict = utils.jload(data_path)
            list_data_dict = datasets.load_from_disk(data_path)
            logging.warning("Formatting inputs...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

            def process_query(example):
                example["instruction"] = example.get("queries", "")
                if example["instruction"] != "":
                    example["instruction"] = example["instruction"].split("<SEP>")[0].strip()
                example["passage"] = example.get("chunks", "")
                example["query"] = example.get("queries", "").split("<SEP>")[1].strip()
                return example

            list_data_dict = list_data_dict.map(process_query)
            sources = [
                prompt_input.format_map(example) if example.get("instruction",
                                                                "") != "" else prompt_no_input.format_map(example)
                for example in tqdm(list_data_dict)
            ]
            targets = [example['labels'] for example in list_data_dict]

            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)
            logging.warning("Saving tokenized data...")
            tokenized_data_path = data_path + "_tokenized.pkl"
            pickle.dump(data_dict, open(tokenized_data_path, "wb"))
        else:
            logging.warning("Skip tokenizing, Loading tokenized data...")
            data_dict = pickle.load(open(data_path + "_tokenized.pkl", "rb"))

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

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

CUSTOM_TOPICS = {
    'dl21': '/mnt/ceph_home/data/trec/data/test_21/2021_queries.tsv',
    'dl22': '/mnt/ceph_home/data/trec/data/test_22/2022_queries.tsv'
}

CUSTOM_QRELS = {
    'dl21': '/mnt/ceph_home/data/trec/data/test_21/2021.qrels.pass.final.txt',
    'dl22': '/mnt/ceph_home/data/trec/data/test_22/2022.qrels.pass.withDupes.txt'
}

CUSTOM_INIT_RUN = {
    'dl21': '/mnt/ceph_home/data/trec/runs/test_21/run.msmarco-v2-passage.unicoil-otf.dl21.txt',
    'dl22': '/mnt/ceph_home/data/trec/runs/test_22/run.msmarco-v2-passage.unicoil-otf.dl22.txt'
}

CUSTOM_INDEX = {
    'dl21': '/mnt/ceph_home/data/trec/lucene-index.msmarco-v2-passage.20220808.4d6d2a.eacd8556dd416ccad517b5e7dc97bceb',
    'dl22': '/mnt/ceph_home/data/trec/lucene-index.msmarco-v2-passage.20220808.4d6d2a.eacd8556dd416ccad517b5e7dc97bceb'
}

from pyserini.search import LuceneSearcher, get_topics, get_qrels
from trec_eval import EvalFunction
from rank_llama import run_retriever, write_eval_file, get_custom_qrels, get_custom_topics, get_formatted_query_passage
import tempfile
import random
import numpy as np
import torch.nn.functional as F

eval_instructions = {
"dl20": [
    "Retrieve a web paragraph that answers the following."
    ],
"dl19": [
    "Retrieve a web paragraph that answers the following."
    ],
"dl21": [
    "Retrieve a web paragraph that answers the following."
    ],
"dl22": [
    "Retrieve a web paragraph that answers the following."
    ],
"touche": [
    "You have to retrieve an argument to this debate question.",
    ],
"dbpedia": [
    "Retrieve a Wikipedia introduction paragraph of the following entity.",
    ],
"news": [
    "Retrieve news articles that provide important context or background information.",
    ],
"scifact": [
    "Retrieve a scientific paper sentence to verify if the following claim is true.",
    ],
"covid": [
    "Retrieve Scientific paper paragraph to answer this question.",
    ],
"nfc": [
    "Retrieve Scientific paper paragraph to answer this question.",
    ],
"signal": [
    "Retrieve tweets that are relevant to the following topics.",
    ],
"robus04": [
    "Retrieve news that are relevant to the following queries.",
    ],
}


class RerankDevDataset(Dataset):
    """complete initial ranking, prepare for reranking"""
    def __init__(self,  tokenizer: transformers.PreTrainedTokenizer, data_name:str, use_custom_init:bool, output_dir):
        super(RerankDevDataset, self).__init__()
        if data_name in THE_INDEX and not use_custom_init:
            print(f"evaluate data {data_name}, and NOT use custom init.")
            index_name = THE_INDEX[data_name]
            searcher = LuceneSearcher.from_prebuilt_index(index_name)
            topics = get_topics(THE_TOPICS[data_name] if data_name != 'dl20' else 'dl20')
            qrels = get_qrels(THE_TOPICS[data_name])
            if index_name == 'msmarco-v2-passage':
                text_is_passage = True
            else:
                text_is_passage = False
            initial_ranking = run_retriever(topics, searcher, qrels, k=100, text_is_passage=text_is_passage)
        elif data_name in CUSTOM_INDEX and use_custom_init:
            print(f"evaluate data {data_name}, and use custom init {use_custom_init}.")
            index_name = CUSTOM_INDEX[data_name]
            searcher = LuceneSearcher(index_name)
            qrels = get_custom_qrels(CUSTOM_QRELS[data_name])
            topics = get_custom_topics(CUSTOM_TOPICS[data_name], qrels)
            initial_ranking = get_formatted_query_passage(searcher, CUSTOM_INIT_RUN[data_name], topics)
        else:
            raise NotImplementedError(f"Dataset {data_name}  and use_custom_init {use_custom_init} not implemented")
        # initail ranking is a list
        # initail ranking[i] = [{'query': query,
        #                        'hits': [{
        #                                  'content': content,
        #                                  'qid': qid,
        #                                  'docid': docid,
        #                                  'rank': rank,
        #                                  'score': score
        #                                  }, ...]
        #                                  }, ...]
        temp_file_bm25 = tempfile.NamedTemporaryFile(dir=output_dir, prefix=f'init_{use_custom_init}_{data_name}', delete=False).name
        write_eval_file(initial_ranking, temp_file_bm25)
        if data_name in THE_TOPICS:
            EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data_name], temp_file_bm25])
        else:
            EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', CUSTOM_QRELS[data_name], temp_file_bm25], trunc=False)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = []
        qids = []
        docids = []
        instruction = eval_instructions[data_name][0]
        for i in range(len(initial_ranking)):
            query = initial_ranking[i]['query']
            for hit in initial_ranking[i]['hits']:
                sources.append(prompt_input.format(query=query, instruction=instruction, passage=hit['content']))
                qids.append(hit['qid'])
                docids.append(hit['docid'])
        ids = {'qids': qids, 'docids': docids}
        logging.warning("Tokenizing inputs...")
        data_dict = preprocess(sources, ids, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = [0] * len(self.input_ids)
        self.qids = data_dict['labels']["qids"]
        self.docids = data_dict['labels']["docids"]
        # save qids and docids for evaluation, temp file will be deleted after evaluation
        self.temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        pickle.dump({'qids': self.qids, 'docids': self.docids}, self.temp_file)
        self.temp_file.close()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        labels = self.labels[idx]
        return dict(
            input_ids=input_ids,
            labels=labels,
        )

@dataclass
class DataCollatorForRerankDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def write_compute_metrics(out_path, id_path, data_name):

    def compute_metrics(eval_preds):
        # read temp file
        with open(id_path.name, 'rb') as f:
            ids = pickle.load(f)
        qids = ids['qids']
        docids = ids['docids']
        # eval_preds is a tuple
        # eval_preds[0] is a list of predictions
        # eval_preds[1] is a list of labels
        predictions, labels = eval_preds
        # take the softmax probability of label 1 as the prediction score for reranking
        predictions = F.softmax(torch.from_numpy(predictions), dim=1)[:, 1].tolist()
        # for each qid rank docids according to predict score
        qid2docid2score = defaultdict(dict)
        for i in range(len(predictions)):
            qid2docid2score[qids[i]][docids[i]] = predictions[i]
        for qid in qid2docid2score:
            qid2docid2score[qid] = sorted(qid2docid2score[qid].items(), key=lambda x: x[1], reverse=True)

        # write predictions to temp file
        with open(out_path, 'w') as f:
            for qid in qid2docid2score:
                for rank, item in enumerate(qid2docid2score[qid]):
                    f.write(f'{qid} Q0 {item[0]} {rank+1} {item[1]} rerank\n')
        # compute metrics
        if data_name in THE_TOPICS:
            ndcg_10 = EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data_name], out_path])
            ndcg_5 = EvalFunction.eval(['-c', '-m', 'ndcg_cut.5', THE_TOPICS[data_name], out_path])
            ndcg_100 = EvalFunction.eval(['-c', '-m', 'ndcg_cut.100', THE_TOPICS[data_name], out_path])
        else:
            ndcg_10 = EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', CUSTOM_QRELS[data_name], out_path], trunc=False)
            ndcg_5 = EvalFunction.eval(['-c', '-m', 'ndcg_cut.5', CUSTOM_QRELS[data_name], out_path], trunc=False)
            ndcg_100 = EvalFunction.eval(['-c', '-m', 'ndcg_cut.100', CUSTOM_QRELS[data_name], out_path], trunc=False)



        metric_dict = {
            'ndcg_10': ndcg_10,
            'ndcg_5': ndcg_5,
            'ndcg_100': ndcg_100
        }
        return metric_dict

    return compute_metrics

