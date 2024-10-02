import sys
sys.path.append('../')

import json
import argparse
from datasets import load_dataset
from load_my_dataset import load_longfact
from utils.base_inference import BaseGenerator, GeneratorWithVD
from utils.PromptTemplateLoader import *
from transformers import GenerationConfig
from tqdm import tqdm
from inference_methods import *

prompt_loader = PromptTemplateLoader()

parser = argparse.ArgumentParser()
parser.add_argument("--sample_responses", action="store_true")
parser.add_argument("--sample_num", type=int, default=16)
parser.add_argument("--benchmark", type=str, default="longfact")
parser.add_argument("--max_sample_num", type=int, default=120)
parser.add_argument("--data_split", type=str, default="test")
parser.add_argument("--voting_num", type=int, default=16)
parser.add_argument("--method", type=str, default="vanilla_llm")
parser.add_argument('--base_model', type=str, default='llama3')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--max_length', type=int, default=4096)
parser.add_argument('--top_p', type=float, default=0.0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument("--device_auto_map", action="store_true")

args = parser.parse_args()

if args.device_auto_map:
    args.device = "auto"
else:
    args.device = f"cuda:{args.device}"

# if args.generation_config:
#     args.generation_config = json.load(open(args.generation_config))
if args.output_path is None:
    if args.method == "voting_decoding" or args.method == "usc" or args.method == "self_reflection":
        args.output_path = f"output/{args.base_model}_{args.method}_voting_{args.voting_num}_{args.data_split}.json"
    else:
        args.output_path = f"output/{args.base_model}_{args.method}_{args.data_split}.json"

args.prompt_dir = "prompt_templates/"
args.standard_prompt_key = "zero_shot"
args.usc_prompt_key = "usc_template"
args.self_reflection_prompt_key = "self_reflection_template"
args.voting_decoding_prompt_key = "voting_decoding_template"

print(args)


if __name__ == "__main__":
    data = load_longfact(max_sample_num=args.max_sample_num)
    questions = [sample["prompt"] for sample in data]
    if args.sample_responses:
        sample_responses(questions, args)
    else:
        inference(questions, args)