import sys
sys.path.append('../')

from inference_methods import *

prompt_loader = PromptTemplateLoader()

parser = argparse.ArgumentParser()
parser.add_argument("--sample_responses", action="store_true")
parser.add_argument("--sample_num", type=int, default=16)
parser.add_argument("--benchmark", type=str, default="truthfulqa")
parser.add_argument("--max_sample_num", type=int, default=None)
parser.add_argument("--data_split", type=str, default="test")
parser.add_argument("--voting_num", type=int, default=16)
parser.add_argument("--method", type=str, default="vanilla_llm")
parser.add_argument('--base_model', type=str, default='llama3')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=160)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--max_length', type=int, default=2048)
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

def load_truthfulqa(max_sample_num=817, split=args.data_split):
    # validation 410:, test :410
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


if __name__ == "__main__":
    data = load_truthfulqa(max_sample_num=args.max_sample_num)
    questions = [sample["question"] for sample in data]
    if args.sample_responses:
        sample_responses(questions, args)
    else: 
        inference(questions, args)

