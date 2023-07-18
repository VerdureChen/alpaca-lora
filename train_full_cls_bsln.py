#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
import datasets
from tqdm import tqdm
import pickle
import os
from full_data import SupervisedDataset, DataCollatorForSupervisedDataset, RerankDevDataset, \
    DataCollatorForRerankDataset, \
    write_compute_metrics
import sys

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path_or_name: str = field(default=None,
                                        metadata={"help": "Path to the evaluation data, or name of the dataset."})
    task_name: Optional[str] = field(default="bsln")
    use_custom_init: Optional[str] = field(default="bm25")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class InferenceArguments:
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    inference_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "The dtype to use for inference."},
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        # output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, output_dir,
                                is_training=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if is_training:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    else:
        train_dataset = None
    if data_args.task_name == "bsln":
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path_or_name)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif data_args.task_name == "rerank":
        eval_dataset = RerankDevDataset(tokenizer=tokenizer, data_name=data_args.eval_data_path_or_name,
                                        use_custom_init=data_args.use_custom_init,
                                        output_dir=output_dir)
        data_collator = DataCollatorForRerankDataset(tokenizer=tokenizer)
    else:
        raise ValueError(f"Task name {data_args.task_name} not recognized.")

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train(model_args, data_args, training_args):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,  # number of classes
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                                              output_dir=training_args.output_dir)
    if data_args.task_name == "bsln":
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    elif data_args.task_name == "rerank":
        outpath = os.path.join(training_args.output_dir, "rerank_results.txt")
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module,
                          compute_metrics=write_compute_metrics(outpath, data_module["eval_dataset"].temp_file,
                                                                data_args.eval_data_path_or_name))
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


def inference(model_args, data_args, training_args, inference_args):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        training_args.output_dir,
        num_labels=2,  # number of classes
        cache_dir=training_args.cache_dir,
        load_in_8bit=inference_args.load_in_8bit,
        torch_dtype=inference_args.inference_dtype,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.output_dir,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                                              output_dir=training_args.output_dir, is_training=False)
    if data_args.task_name == "bsln":
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    elif data_args.task_name == "rerank":
        outpath = os.path.join(training_args.output_dir, f"rerank_results_{data_args.eval_data_path_or_name}.txt")
        print(f"Writing results to {outpath}")
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module,
                          compute_metrics=write_compute_metrics(outpath, data_module["eval_dataset"].temp_file,
                                                                data_args.eval_data_path_or_name))
    trainer.evaluate()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()
    print(f'model_args:{model_args}')
    print(f'data_args:{data_args}')
    print(f'training_args:{training_args}')
    print(f'inference_args:{inference_args}')
    if training_args.do_eval is not True:
        train(model_args, data_args, training_args)
    else:
        inference(model_args, data_args, training_args, inference_args)

