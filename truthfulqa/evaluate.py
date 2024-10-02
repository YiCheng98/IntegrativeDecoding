import json
import sys
sys.path.append('..')
sys.path.append('../..')
import os
import argparse
from utils.PromptTemplateLoader import PromptTemplateLoader
from datasets import load_dataset
from utils.call_LLM_API import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_split", type=str, default="test")
parser.add_argument("--eval_all_files", action="store_true")
parser.add_argument("--key_word", type=str, default=None)
parser.add_argument("--eval_dir", type=str, default="output/")
parser.add_argument("--eval_data_path", type=str, default="output/glm_vanilla_llm_test.json")
parser.add_argument("--no_informativeness", action="store_true")

args = parser.parse_args()
prompt_loader = PromptTemplateLoader(template_dir_path="prompt_templates/")



print(args)

def load_truthfulqa(max_sample_num=817, split=args.data_split):
    # validation 0:400, test 400:817
    ori_dataset = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    dataset = []
    index = -1
    for sample in ori_dataset:
        index += 1
        question = sample["question"]
        correct_answer = sample['correct_answers']
        incorrect_answers = sample['incorrect_answers']
        dataset.append({
            "question": question,
            "correct_answers": correct_answer,
            "incorrect_answers": incorrect_answers
        })
    if split=="test":
        dataset = dataset[:410]
    elif split == "validation":
        dataset = dataset[410:]
    else:
        raise ValueError(f"Split {split} not supported")
    if max_sample_num is not None:
        dataset = dataset[:max_sample_num]
    return dataset


def eval_truthful(data):
    messages = []
    prompt_key = "eval_truth"
    for sample in data:
        response = sample["response"].strip()
        response = response[response.rfind("\n") + 1:]
        if response.lower().startswith("**refined answer:**"):
            response = response[len("**Refined Answer:**"):]
        # print(response)
        # if response.lower().find("refined answer") != -1:
        #     response = response[response.lower().find("refined answer") + len("refined answer"):]
        #     response = response[response.find("\n") + 1:]
        #     print(response)
        #     print("==="*10)
        template_placeholder = {"question": sample["question"],
                                "correct_answers": "\n".join(sample["correct_answers"]),
                                "incorrect_answers": "\n".join(sample["incorrect_answers"]),
                                "generated_answer": response}
        prompt = prompt_loader.construct_prompt(prompt_key, template_placeholder)
        messages.append([{"role": "user", "content": prompt}])
    eval_results = asyncio.run(call_LLM_async(messages, model="gpt-4",
                                              batch_size=16,
                                              tqdm_visible=True))
    cnt = {"correct": 0, "wrong": 0}
    for i in range(len(data)):
        eval_result = eval_results[i].strip().lower()
        # remove all the characters in the beginning of the string that are not letters
        for j in range(len(eval_result)):
            if eval_result[j].isalpha():
                eval_result = eval_result[j:]
                break
        if eval_result.startswith("correct"):
            cnt["correct"] += 1
        elif eval_result.startswith("wrong"):
            cnt["wrong"] += 1
    total_valid = cnt["correct"] + cnt["wrong"]
    return cnt['correct'] / total_valid


def eval_informativeness(data):
    if args.no_informativeness:
        return 1.0
    messages = []
    prompt_key = "eval_info"
    for sample in data:
        template_placeholder = {"question": sample["question"],
                                "answer": sample["response"].strip()}
        prompt = prompt_loader.construct_prompt(prompt_key, template_placeholder)
        messages.append([{"role": "user", "content": prompt}])
    eval_results = asyncio.run(call_LLM_async(messages, model="gpt-4",
                                              batch_size=16,
                                              tqdm_visible=True))
    cnt = {"complete": 0, "incomplete": 0}
    for i in range(len(data)):
        eval_result = eval_results[i].strip().lower()
        # remove all the characters in the beginning of the string that are not letters
        for j in range(len(eval_result)):
            if eval_result[j].isalpha():
                eval_result = eval_result[j:]
                break
        if eval_result.startswith("yes"):
            cnt["complete"] += 1
        elif eval_result.startswith("no"):
            cnt["incomplete"] += 1
    total_valid = cnt["complete"] + cnt["incomplete"]
    return cnt['complete'] / total_valid

def eval_single_file(file_path):
    response_data = json.load(open(file_path, "r"))
    data_len = len(response_data)
    data = load_truthfulqa(split=args.data_split)
    data = data[:data_len]

    for i in range(data_len):
        data[i]["response"] = response_data[i]["generated_answer"]

    truthful_score = eval_truthful(data)
    info_score = eval_informativeness(data)
    t_times_i = truthful_score * info_score
    return {"truthful": truthful_score, "informativeness": info_score, "t_times_i": t_times_i}

if __name__ == "__main__":
    if args.eval_all_files:
        eval_result_file = "eval_results.txt"
        results = ""
        files = os.listdir(args.eval_dir)
        finished_eval_files = []
        if os.path.exists(eval_result_file):
            with open(eval_result_file, "r") as f:
                previous_results = f.read()
                finished_eval_files = [line[:line.find(".json")+5] for line in previous_results.split("\n") if line]
            results = previous_results
        for file in files:
            if args.key_word is not None and file.find(args.key_word) == -1:
                continue
            if file.endswith(".json"):
                if file in finished_eval_files:
                    print(f"Skipping {file}")
                    continue
                print(f"Evaluating {file}")
                file_path = os.path.join(args.eval_dir, file)
                result = eval_single_file(file_path)
                print(result)
                results += f"{file}: {json.dumps(result)}\n"
                with open(eval_result_file, "w") as f:
                    f.write(results)
        lines = [line for line in results.split("\n") if line]
        lines.sort()
        with open(eval_result_file, "w") as f:
            f.write("\n".join(lines)+"\n")
    else:
        print(f"Evaluating {args.eval_data_path}")
        result = eval_single_file(args.eval_data_path)
        print(result)










