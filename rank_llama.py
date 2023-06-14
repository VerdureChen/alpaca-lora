import copy
from tqdm import tqdm
import torch
import json


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True



def num_tokens_from_messages(tokenizer, messages, max_tokens=2048, return_tokenized=False):
    """Returns the number of tokens."""
    passage_tokenized = tokenizer(
        [messages],
        truncation=False,
        max_length=max_tokens,
        padding=False,
        return_tensors=None,
    )['input_ids']

    num_tokens = sum(len(x) for x in passage_tokenized)
    return num_tokens if not return_tokenized else (num_tokens, passage_tokenized)


def create_permutation_instruction(item=None, rank_start=0, rank_end=100, prompter=None, tokenizer=None):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300
    passage_lst = []
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        # content = content.replace('Title: Content: ', '')
        # content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        # content = ' '.join(content.split()[:int(max_length)])
        passage_lst.append({'rank': rank, 'content': content})
    input_text = '\n'.join([f"[{item['rank']}] {item['content']}" for item in passage_lst])
    full_prompt = prompter.generate_prompt(
        query,
        input_text,
        num=num,
    )
    prompt_without_input = prompter.generate_prompt(
        query,
        '',
        num=num,
    )
    if num_tokens_from_messages(tokenizer, full_prompt) > 2000:
        prompt_without_length = num_tokens_from_messages(tokenizer, prompt_without_input)
        average_passage_length = int((2000 - prompt_without_length) / num)
        for passage in passage_lst:
            passage_tokenized = num_tokens_from_messages(tokenizer, passage['content'], return_tokenized=True)
            passage_token_num = passage_tokenized[0]
            if passage_token_num > average_passage_length:
                passage['content'] = tokenizer.decode(passage_tokenized[1][0][:average_passage_length])
                print(f'query is {query}, passage_rank is {passage["rank"]}')
                print(f'the length of the truncated passage is {average_passage_length}')
                print(f'the length of the original passage is {passage_token_num}')
    input_text = '\n'.join([f"[{item['rank']}] {item['content']}" for item in passage_lst])
    print(f'the length of the input text is {num_tokens_from_messages(tokenizer, input_text)}')
    full_prompt = prompter.generate_prompt(
        query,
        input_text,
        num=num,
    )

    return full_prompt


def run_llm(messages, model=None, tokenizer=None, generate_config=None, device='cuda', prompter=None):
    inputs = tokenizer(messages, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generate_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdecimal():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model=None, tokenizer=None, prompter=None,
                         generate_config=None, device='cuda'):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, prompter=prompter, tokenizer=tokenizer)
    # print(f'messages is {messages}')
    permutation = run_llm(messages, model=model, tokenizer=tokenizer, generate_config=generate_config, device=device, prompter=prompter)
    # print(f'permutation is {permutation}')
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    # print(f'item is {item}')
    return item


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10,
                    model=None, tokenizer=None, prompter=None, generate_config=None,
                    device='cuda'):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        # print(f'start_pos is {start_pos}, end_pos is {end_pos}')
        # print(f'item is {item}')
        item = permutation_pipeline(item, start_pos, end_pos,
                                    model=model, tokenizer=tokenizer,
                                    prompter=prompter, generate_config=generate_config, device=device)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def main():
    from pyserini.search import LuceneSearcher
    from pyserini.search import get_topics, get_qrels
    import tempfile

    openai_key = None  # Your openai key

    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    topics = get_topics('dl19-passage')
    qrels = get_qrels('dl19-passage')

    rank_results = run_retriever(topics, searcher, qrels, k=100)

    new_results = []
    for item in tqdm(rank_results):
        new_item = permutation_pipeline(item, rank_start=0, rank_end=10,)
        new_results.append(new_item)

    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(new_results, temp_file)
    from trec_eval import EvalFunction

    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', 'dl19-passage', temp_file])


if __name__ == '__main__':
    main()