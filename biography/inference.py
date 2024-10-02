import sys
sys.path.append('../')

import json
import argparse
from datasets import load_dataset
from utils.base_inference import BaseGenerator, GeneratorWithVD
from utils.PromptTemplateLoader import *
from transformers import GenerationConfig
from tqdm import tqdm
from inference_methods import *

prompt_loader = PromptTemplateLoader()

parser = argparse.ArgumentParser()
parser.add_argument("--sample_responses", action="store_true")
parser.add_argument("--sample_num", type=int, default=16)
parser.add_argument("--benchmark", type=str, default="biography")
parser.add_argument("--max_sample_num", type=int, default=None)
parser.add_argument("--data_split", type=str, default="test")
parser.add_argument("--voting_num", type=int, default=16)
parser.add_argument("--method", type=str, default="vanilla_llm")
parser.add_argument('--base_model', type=str, default='llama3')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--top_p', type=float, default=0.0)
parser.add_argument('--max_length', type=int, default=2048)
parser.add_argument('--device', type=int, default=0)
parser.add_argument("--device_auto_map", action="store_true")

args = parser.parse_args()

if args.device_auto_map:
    args.device = "auto"
else:
    args.device = f"cuda:{args.device}"

if args.output_path is None:
    if args.method == "voting_decoding" or args.method == "usc" or args.method == "self_reflection":
        args.output_path = f"output/{args.base_model}_{args.method}_voting_{args.voting_num}_{args.data_split}.json"
        if args.temperature != 0.0 or args.top_p != 0.0:
            args.output_path = args.output_path[:-5] + f"_temperature_{args.temperature}_top_p_{args.top_p}.json"
    else:
        args.output_path = f"output/{args.base_model}_{args.method}_{args.data_split}.json"

args.prompt_dir = "prompt_templates/"
args.standard_prompt_key = "gen_biography"
args.usc_prompt_key = "usc_template"
args.self_reflection_prompt_key = "self_reflection_template"
args.voting_decoding_prompt_key = "voting_decoding_template"

print(args)
BIOGRAPHY_DATA_PATH = "dataset/article_200.json"
def load_biography(max_sample_num=128, split=args.data_split):
    # test 0:128
    # validation 128:
    with open(BIOGRAPHY_DATA_PATH, "r") as f:
        data = json.load(f)
    names = list(data.keys())
    print(len(names))
    dataset = []
    for name in names:
        dataset.append({
            "question": name,
        })
    if split=="test":
        dataset = dataset[:128]
    elif split == "validation":
        dataset = dataset[128:]
    else:
        raise ValueError(f"Split {split} not supported")
    if max_sample_num is not None:
        dataset = dataset[:max_sample_num]
    return dataset


if __name__ == "__main__":
    data = load_biography(max_sample_num=args.max_sample_num)
    questions = [sample["question"] for sample in data]
    if args.sample_responses:
        sample_responses(questions, args)
    else: 
        inference(questions, args)
