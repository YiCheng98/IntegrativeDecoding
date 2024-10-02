import sys
sys.path.append("../")

import time
import asyncio
import json
import os
from tqdm import tqdm
import argparse
import itertools
from utils.call_LLM_API import call_LLM_async
_SENTENCE = 'sentence'
_ATOMIC_FACTS = 'atomic_facts'

parser = argparse.ArgumentParser()
parser.add_argument("--max_sample_num", type=int, default=120)
parser.add_argument("--eval_all_files", action="store_true")
parser.add_argument("--key_word", type=str, default=None)
parser.add_argument("--eval_dir", type=str, default="output/atomic_facts")
parser.add_argument("--file_name", type=str, default="qwen_vanilla_llm_test.json")

args = parser.parse_args()
print(args)
# pylint: enable=g-bad-import-order

def convert_atomic_facts_to_dicts(
    outputted_facts: list[tuple[str, list[str]]]
):
  return [
      {_SENTENCE: sentence, _ATOMIC_FACTS: identified_atomic_facts}
      for sentence, identified_atomic_facts in outputted_facts
  ]

def parse_yes_no(text: str):
    """Parse a yes/no response."""
    text = text.lower().strip()
    if text.startswith("yes"):
        return True
    else:
        # print(text)
        return False

def verify_with_gpt4(prompt, response, extracted_fact_info):
    atomic_facts = []
    for facts in extracted_fact_info['all_atomic_facts']:
        for fact in facts['atomic_facts']:
            atomic_facts.append(fact)
    sentence_and_atomic_facts_list = extracted_fact_info['sentences_and_atomic_facts']
    sentences = []
    atomic_facts_in_one_sentence_list = []
    for sentence_and_atomic_facts in sentence_and_atomic_facts_list:
        sentences.append(sentence_and_atomic_facts[0])
        atomic_facts_in_one_sentence_list.append(sentence_and_atomic_facts[1])

    verify_sentence_prompt_template = ("{context}\n\nRead the above text carefully. Note that some of the information in it might be incorrect. \nIs the sentence \"{sentence}\" in the above text factual and correct? Your response should either \"Yes\" or \"No\".")
    verify_fact_prompt_template = ("{context}\n\nRead the above text carefully. Note that some of the information in it might be incorrect. \nIs the claim \"{fact}\" in the sentence \"{sentence}\" factual and correct?\n Your response should either \"Yes\" or \"No\".")
    # verify at the sentence level
    verify_prompts = []
    # print("Start verifying with GPT-4 ...")
    for i, sentence in enumerate(sentences):
        # context = prompt + "\n" + response
        verify_prompts.append(verify_sentence_prompt_template.format(sentence=sentence, context=prompt + "\n" + response))
        # print(verify_prompts[-1])
        # print("-"*50)
    messages =[[{"role": "user", "content": verify_prompt}] for verify_prompt in verify_prompts]
    verify_responses = asyncio.run(call_LLM_async(messages, model="gpt-4", batch_size=16, tqdm_visible=False))
    # verify_responses = []
    sentences_verify_results = [parse_yes_no(r) for r in verify_responses]
    # print(sentences_verify_results)
    # if the verify result of a sentence is no, then further verify the atomic facts in the sentence
    verify_prompts = []
    for i, atomic_facts_in_one_sentence in enumerate(atomic_facts_in_one_sentence_list):
        if not sentences_verify_results[i]:
            # context = sentences[max(0, i-5):min(i+1, len(sentences))]
            for fact in atomic_facts_in_one_sentence:
                verify_prompts.append(verify_fact_prompt_template.format(fact=fact, sentence=sentences[i], context=prompt + "\n" + response))
    messages =[[{"role": "user", "content": verify_prompt}] for verify_prompt in verify_prompts]
    verify_responses = asyncio.run(call_LLM_async(messages, model="gpt-4", batch_size=16, tqdm_visible=False))
    # verify_responses = []
    atomic_facts_verify_results = [parse_yes_no(r) for r in verify_responses]

    verify_results = []
    for i, atomic_facts_in_one_sentence in enumerate(atomic_facts_in_one_sentence_list):
        if sentences_verify_results[i]:
            verify_results.extend([True] * len(atomic_facts_in_one_sentence))
        else:
            verify_results.extend(atomic_facts_verify_results[:len(atomic_facts_in_one_sentence)])
            atomic_facts_verify_results = atomic_facts_verify_results[len(atomic_facts_in_one_sentence):]

    cnt_correct = sum(verify_results)
    cnt_wrong = len(verify_results) - cnt_correct
    prec = cnt_correct / len(verify_results)
    recall_64 = min(cnt_correct / 64, 1)
    recall_128 = min(cnt_correct / 128, 1)
    recall_178 = min(cnt_correct / 178, 1)
    f1_64 = 0 if prec + recall_64 == 0 else 2 * prec * recall_64 / (prec + recall_64)
    f1_128 = 0 if prec + recall_128 == 0 else 2 * prec * recall_128 / (prec + recall_128)
    f1_178 = 0 if prec + recall_178 == 0 else 2 * prec * recall_178 / (prec + recall_178)
    detailed_eval_results = {}
    for i, fact in enumerate(atomic_facts):
        detailed_eval_results[fact] = verify_results[i]
    return {
        "detailed_eval_results": detailed_eval_results,
        "supported": cnt_correct,
        "unsupported": cnt_wrong,
        "precision": prec,
        "recall@64": recall_64,
        "recall@128": recall_128,
        "recall@178": recall_178,
        "f1@64": f1_64,
        "f1@128": f1_128,
        "f1@178": f1_178
    }


def eval_one_file(file_name, eval_data_path, max_sample_num, eval_result_dir="eval_results/"):
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)
    eval_result_path = os.path.join(eval_result_dir, eval_data_path.split("/")[-1])
    if os.path.exists(eval_result_path):
        with open(eval_result_path, "r") as f:
            eval_results = json.load(f)
    else:
        eval_results = []
    with open(eval_data_path, "r") as f:
        data = json.load(f)
    data = data[len(eval_results):]
    for sample in tqdm(data[:max_sample_num], desc="Annotation Progress"):
        prompt = sample["question"]
        response = sample["generated_answer"]
        extracted_fact_info = sample["extracted_fact_info"]
        eval_results.append(verify_with_gpt4(prompt, response=response, extracted_fact_info=extracted_fact_info))
        with open(eval_result_path, "w") as f:
            json.dump(eval_results, f)
        # print(json.dumps(eval_results[-1], indent=4))

    avg_result = {
        "precision": sum([result["precision"] for result in eval_results]) / len(eval_results),
        "recall@64": sum([result["recall@64"] for result in eval_results]) / len(eval_results),
        "recall@128": sum([result["recall@128"] for result in eval_results]) / len(eval_results),
        "recall@178": sum([result["recall@178"] for result in eval_results]) / len(eval_results),
        "f1@64": sum([result["f1@64"] for result in eval_results]) / len(eval_results),
        "f1@128": sum([result["f1@128"] for result in eval_results]) / len(eval_results),
        "f1@178": sum([result["f1@178"] for result in eval_results]) / len(eval_results),
    }
    with open(eval_result_path, "w") as f:
        json.dump(eval_results, f)
    print(avg_result)

    with open("eval_results/result_summary.txt", "r") as f:
        lines = f.read().split('\n')
    for line in lines:
        if line.startswith(f"{file_name}:"):
            lines.remove(line)
    lines.append(f"{file_name}: {json.dumps(avg_result)}")
    lines.sort()
    lines = [line for line in lines if len(line) > 0]
    with open("eval_results/result_summary.txt", "w") as f:
        f.write("\n".join(lines))
    if len(data) > 1:
        print("Sleep for 120 seconds ...")
        time.sleep(120)



if __name__ == '__main__':
    if not os.path.exists("eval_results"):
        os.makedirs("eval_results")
    if not os.path.exists("eval_results/result_summary.txt"):
        with open("eval_results/result_summary.txt", "w") as f:
            f.write("")
    if args.eval_all_files:
        file_name_list = os.listdir(args.eval_dir)
        if args.key_word:
            file_name_list = [file_name for file_name in file_name_list if file_name.find(args.key_word) != -1]
        for file_name in tqdm(file_name_list, desc="File Progress"):
            print(f"Start evaluating {file_name} ...")
            eval_data_path = os.path.join(args.eval_dir, file_name)
            try:
                eval_one_file(file_name=file_name,
                              eval_data_path=eval_data_path,
                              max_sample_num=args.max_sample_num)
            except:
                print(f"Error occurs when evaluating {file_name}.")
    else:
        eval_data_path = os.path.join(args.eval_dir, args.file_name)
        max_sample_num = args.max_sample_num
        eval_one_file(file_name=args.file_name,
                      eval_data_path=eval_data_path,
                      max_sample_num=args.max_sample_num)

