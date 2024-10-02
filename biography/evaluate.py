import json
import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
from tqdm import tqdm
import copy
import asyncio
import os
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--use_gpt4", action="store_true")
parser.add_argument("--eval_all_files", action="store_true")
parser.add_argument("--key_word", type=str, default=None)
parser.add_argument("--eval_dir", type=str, default="output/")
parser.add_argument("--eval_data_path", type=str, default="output/gemma_voting_decoding_voting_12_test.json")
# parser.add_argument("--eval_data_path", type=str, default="output/llama3_vanilla_llm_test.json")

args = parser.parse_args()

# # from utils.call_LLM_API import *
# args.use_gpt4 = True
# if args.use_gpt4:
from utils.call_LLM_API import *

print(args)

def parse_bullets(sentence):
    sentence = sentence.replace('*','')
    if sentence.find("Question 2") != -1:
        sentence = sentence[sentence.find("Question 2"):]
    if sentence.lower().find("refined answer") != -1:
        sentence = sentence[sentence.lower().find("refined answer") + len("refined answer"):]
        sentence = sentence[sentence.find("\n") + 1:]
        # print(sentence)
    bullets_preprocess = [l for l in sentence.split("\n") if len(l) > 0]
    if len(bullets_preprocess) == 1:
        bullets_preprocess = [l+'.' for l in sentence.split(".") if len(l) > 0]
    new_bullets_preprocess = []
    for bullet in bullets_preprocess:
        if (bullet.find("Here is") != -1 or
                bullet[-1] == ':' or
                bullet.find("Here are") != -1 or
                bullet.find("I apologize") != -1 or
                bullet.find("Sorry") != -1 or
                bullet.find("Thank you") != -1 or
                bullet.find("Please") != -1 or
                bullet.find("I hope")) != -1:
            continue
        new_bullets_preprocess.append(bullet)
    bullets_preprocess = new_bullets_preprocess

    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """

    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def filter_people(person):
    people = person.split("(")[0]
    return people


def eval_one_file(eval_data_path):
    print(eval_data_path)
    response_data = json.load(open(eval_data_path, "r"))
    print(f"Evaluating {eval_data_path}")
    with open("dataset/article.json", "r") as f:
        gt_data = json.load(f)

    gt_data_filter = {}

    for k, v in gt_data.items():
        k = filter_people(k)
        gt_data_filter[k] = v

    gt_data = gt_data_filter

    # people = [sample["question"] for sample in response_data]

    accuracies = []
    messages = []
    # messages_gpt4 = []
    total_gt_bullets = 0
    bio_bullets_list = []
    for index, sample in enumerate(response_data):
        person = sample["question"]
        if person.find("(") != -1:
            person = person.split("(")[0]
        response = sample["generated_answer"]
        if person not in gt_data:
            continue
        gt_description = gt_data[person]
        gt_bullets = parse_bullets(gt_description)
        total_gt_bullets += len(gt_bullets)
        gt_bullets = " ".join(gt_bullets)
        # bio_descriptions = parse_bullets(response)  # [2][-1]['content']
        bio_description = response
        bio_bullets = parse_bullets(bio_description)[:5]
        bio_bullets_list.append(bio_bullets)
        # continue

        for bullet in bio_bullets:
            message = [{"role": "user",
                        "content": "Reference: \n {} \n\n Based on the above reference and your own knowledge about the computer scientist {}, is the following statement about the achievement made by this computer scientist correct and factual? \n {} \n Give a single word answer, yes or no. ".format(
                            gt_bullets, person, bullet)}]
            # message = [{"role": "user", "content": "Consider the following fact of {}: \n {} \n Is this fact correct? Give a single word answer, yes or no. ".format(person, bullet)}]
            messages.append(copy.deepcopy(message))
            # messages_gpt4.append(copy.deepcopy(message_gpt4))

    # if args.use_gpt4:
    #     eval_results = asyncio.run(call_LLM_async(messages_gpt4, model="gpt-4",
    #                                               batch_size=16,
    #                                               tqdm_visible=True))
    # else:
    eval_results = asyncio.run(call_LLM_async(messages, model="gpt-4",
                                              batch_size=16,
                                              tqdm_visible=True))

    true_positive = 0
    for i, result in enumerate(eval_results):
        accurate = parse_yes_no(result)
        # if not accurate:
        #     print(messages[i][0]["content"])
        #     print("-"*30)
        if accurate is not None:
            accuracies.append(accurate)
            if accurate:
                true_positive += 1
        else:
            # print("uncertain")
            accuracies.append(False)
    sample_eval_results = []
    index = 0
    for bio_bullets in bio_bullets_list:
        bullet_num = len(bio_bullets)
        tmp = accuracies[index:index + bullet_num]
        if bullet_num== 0:
            sample_eval_results.append({
                "correct_num": 0,
                "incorrect_num": 0,
            })
        else:
            sample_eval_results.append({
                "correct_num": sum(tmp),
                "incorrect_num": bullet_num - sum(tmp),
            })
        index = index + bullet_num

    avg_correct_num = np.mean([sample["correct_num"] for sample in sample_eval_results])
    avg_incorrect_num = np.mean([sample["incorrect_num"] for sample in sample_eval_results])
    avg_accuracy = np.mean(accuracies)
    print(f"correct_num: {avg_correct_num}, incorrect_num: {avg_incorrect_num}, accuracy: {avg_accuracy}")
    return {"correct_num": avg_correct_num, "incorrect_num": avg_incorrect_num, "accuracy": avg_accuracy}


if __name__ == "__main__":
    if not args.eval_all_files:
        eval_one_file(args.eval_data_path)
    else:
        eval_result_path = "eval_results.txt"
        if os.path.exists(eval_result_path):
            with open(eval_result_path, "r") as f:
                eval_results_text = f.read()
        else:
            eval_results_text = ""
        eval_data_dir = args.eval_dir
        eval_data_paths = [os.path.join(eval_data_dir, f) for f in os.listdir(eval_data_dir)]
        eval_data_paths = [f for f in eval_data_paths if f.endswith(".json")]
        if args.key_word:
            eval_data_paths = [f for f in eval_data_paths if f.find(args.key_word) != -1]
        for eval_data_path in eval_data_paths:
            if eval_data_path.endswith(".json"):
                if eval_results_text.find(eval_data_path[len(eval_data_dir):]) != -1:
                    print(f"Skipping {eval_data_path}")
                    continue
                result = eval_one_file(eval_data_path)
                eval_results_text += f"{eval_data_path[len(eval_data_dir):]}: {json.dumps(result)}\n"
                with open("eval_results.txt", "w") as f:
                    f.write(eval_results_text)
        lines = [line for line in eval_results_text.split("\n") if line]
        lines.sort()
        eval_results_text = "\n".join(lines)+"\n"
        with open("eval_results.txt", "w") as f:
            f.write(eval_results_text)





