from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from utils.prompter import Prompter

class AlpacaDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, path_to_data: str, prompt_template_name: str,
                 train_on_inputs: bool = False, add_eos_token: bool = True, cutoff_len: int = 2048,
                 val_set_size: int = 0, split: str = 'train'):
        self.tokenizer = tokenizer
        self.prompter = Prompter(prompt_template_name)
        self.prompt_template_name = prompt_template_name
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.cutoff_len = cutoff_len
        self.val_set_size = val_set_size
        if path_to_data.endswith('.json') or path_to_data.endswith('.jsonl'):
            self.dataset = datasets.load_dataset('json', data_files=path_to_data)['train']
        else:
            self.dataset = datasets.load_from_disk(path_to_data)

        if self.val_set_size > 0:
            self.train_val = self.dataset.train_test_split(test_size=self.val_set_size, shuffle=True, seed=42)
            print(f"Train set size: {len(self.train_val['train'])}, Val set size: {len(self.train_val['test'])}")
            if split == 'val':
                self.dataset = self.train_val['test']
            else:
                self.dataset = self.train_val['train']

    def __len__(self):
        return len(self.dataset)

    def _tokenize(self, example, add_eos_token=True, cutoff_len=2048):
        result = self.tokenizer(
            example,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def __getitem__(self, idx):
        data_point = self.dataset[idx]
        if self.prompt_template_name == "reranker":
            full_prompt = self.prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
                num=data_point["num"],
            )
        elif self.prompt_template_name == "alpaca":
            full_prompt = self.prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        else:
            raise NotImplementedError
        tokenized_full_prompt = self._tokenize(full_prompt, add_eos_token=self.add_eos_token, cutoff_len=self.cutoff_len)

        if not self.train_on_inputs:
            if self.prompt_template_name == "reranker":
                user_prompt = self.prompter.generate_prompt(
                    data_point["instruction"], data_point["input"], num=data_point["num"],
                )
            elif self.prompt_template_name == "alpaca":
                user_prompt = self.prompter.generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
            else:
                raise NotImplementedError

            tokenized_user_prompt = self._tokenize(
                user_prompt, add_eos_token=self.add_eos_token, cutoff_len=self.cutoff_len
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

