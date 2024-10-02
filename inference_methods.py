import sys
import json
import argparse
from datasets import load_dataset
from utils.base_inference import BaseGenerator, GeneratorWithVD
from utils.PromptTemplateLoader import *
from transformers import GenerationConfig
from tqdm import tqdm
import copy

MODEL_PATH_DICT = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma": "google/gemma-2-9b-it",
    "qwen": "Qwen/Qwen2-7B-Instruct",
    "phi": "microsoft/Phi-3-small-8k-instruct",
    "glm": "THUDM/glm-4-9b-chat",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407"
}

def load_sample_responses(benchmark, data_split, base_model, sample_num, max_sample_num, temperature=0.0, top_p=0.0):
    cache_file_dir ="cache_sampled_responses"
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)
        return None
    file_name_pre = f"{benchmark}_{data_split}_{base_model}_"
    # print(f"Loading cache file with prefix {file_name_pre}")
    cache_files = [file_name for file_name in os.listdir(cache_file_dir) if file_name.startswith(file_name_pre)]
    for file_name in cache_files:
        file_sample_num = int(file_name[len(file_name_pre):].split("_")[0])
        if ((file_name.find(f"temperature_{temperature}_") != -1 and file_name.find(f"top_p_{top_p}.") != -1) or (temperature==0 and top_p==0)) and file_sample_num >= sample_num:
            # load the file
            print(f"Find the cache file of sampled responses: {file_name}")
            with open(f"{cache_file_dir}/{file_name}", "r") as f:
                data = json.load(f)
            sampled_responses = [sample["sampled_responses"][:sample_num] for sample in data]
            return sampled_responses[:max_sample_num]
    return None


def inference(questions, args):
    if args.base_model in MODEL_PATH_DICT.keys():
        args.model_path = MODEL_PATH_DICT[args.base_model]
    else:
        raise ValueError(f"Model {args.base_model} not supported")
    if args.method == "vanilla_llm":
        if args.temperature == 0.0:
            # set generation config to be greedy decoding
            generation_config = {"do_sample": False}
        else:
            generation_config = {"do_sample": True, "temperature": args.temperature}
        if args.max_length:
            generation_config["max_length"] = args.max_length
        model = VanillaLLM(model_path = args.model_path, 
                            generation_config=generation_config,
                            prompt_dir=args.prompt_dir,
                            prompt_key=args.standard_prompt_key,
                            device=args.device)
        responses = model.inference_on_dataset(questions, batch_size=args.batch_size)
        output = []
        for i, question in enumerate(questions):
            output.append({
                "question": question,
                "generated_answer": responses[i],
                "question_index": i
            })
    elif args.method == "dola":
        model = DoLaInference(model_path = args.model_path, 
                            prompt_dir=args.prompt_dir,
                            prompt_key=args.standard_prompt_key,
                            device=args.device)
        responses = model.inference_on_dataset(questions, batch_size=args.batch_size)
        output = []
        for i, question in enumerate(questions):
            output.append({
                "question": question,
                "generated_answer": responses[i],
                "question_index": i
            })
    elif args.method == "usc":
        model = USCInference(model_path = args.model_path, 
                            voting_num=args.voting_num,
                            prompt_dir=args.prompt_dir,
                            prompt_key=args.usc_prompt_key,
                            device=args.device)
        sampled_responses = load_sample_responses(benchmark=args.benchmark, 
                                                    data_split=args.data_split, 
                                                    base_model=args.base_model, 
                                                    sample_num=args.voting_num, 
                                                    max_sample_num=args.max_sample_num,
                                                    temperature=args.temperature,
                                                    top_p=args.top_p)
        responses = model.inference_on_dataset(questions, batch_size=args.batch_size, sampled_responses=sampled_responses)
        output = []
        for i, question in enumerate(questions):
            output.append({
                "question": question,
                "generated_answer": responses[i],
                "question_index": i
            })
    elif args.method == "self_reflection":
        model = SelfReflectionInference(model_path=args.model_path, 
                                        voting_num=args.voting_num,
                                        prompt_dir=args.prompt_dir,
                                        prompt_key=args.self_reflection_prompt_key,
                                        device=args.device)
        sampled_responses = load_sample_responses(benchmark=args.benchmark, 
                                                    data_split=args.data_split, 
                                                    base_model=args.base_model, 
                                                    sample_num=args.voting_num, 
                                                    max_sample_num=args.max_sample_num,
                                                    temperature=args.temperature,
                                                    top_p=args.top_p)
        responses = model.inference_on_dataset(questions, batch_size=args.batch_size, sampled_responses=sampled_responses)
        output = []
        for i, question in enumerate(questions):
            output.append({
                "question": question,
                "generated_answer": responses[i],
                "question_index": i
            })
    elif args.method == "voting_decoding":
        model = VotingDecodingInference(voting_num=args.voting_num,
                                        model_path=args.model_path,
                                        prompt_dir=args.prompt_dir,
                                        standard_prompt_key=args.standard_prompt_key,
                                        voting_decoding_prompt_key=args.voting_decoding_prompt_key,
                                        device=args.device)
        sampled_responses = load_sample_responses(benchmark=args.benchmark, 
                                                    data_split=args.data_split, 
                                                    base_model=args.base_model, 
                                                    sample_num=args.voting_num, 
                                                    max_sample_num=args.max_sample_num,
                                                    temperature=args.temperature,
                                                    top_p=args.top_p)
        responses, middle_results = model.inference_on_dataset(questions, batch_size=args.batch_size, sampled_responses=sampled_responses, max_length=args.max_length)
        output = []
        # print(len(responses))
        for i, question in enumerate(questions):
            output.append({
                "question": question,
                "generated_answer": responses[i],
                "question_index": i
            })
    else:
        raise NotImplementedError
    print(args.output_path)
    with open(args.output_path, "w") as f:
        json.dump(output, f)


def sample_responses(questions, args):
    if args.base_model in MODEL_PATH_DICT.keys():
        args.model_path = MODEL_PATH_DICT[args.base_model]
    else:
        raise ValueError(f"Model {args.base_model} not supported")
    
    
    generation_config = {"do_sample": True}
    if  args.temperature == 0.0 and args.top_p == 0.0:
        generation_config["temperature"] = 0.7
    else:
        if args.temperature != 0:
            generation_config["temperature"] = args.temperature
        else:
            generation_config["top_p"] = args.top_p
    generation_config["max_length"]=args.max_length
    if args.sample_responses:
        if not os.path.exists("cache_sampled_responses"):
            os.makedirs("cache_sampled_responses")
        args.output_path = f"cache_sampled_responses/{args.benchmark}_{args.data_split}_{args.base_model}_{args.sample_num}_responses_temperature_{args.temperature}_top_p_{args.top_p}.json"
        # print(args.output_path)
    repeated_questions = []
    for _ in range(args.sample_num):
        for question in questions:
            repeated_questions.append(copy.deepcopy(question))
    model = VanillaLLM(model_path=args.model_path, 
                        generation_config=generation_config,
                        prompt_dir=args.prompt_dir,
                        prompt_key=args.standard_prompt_key,
                        device=args.device)
    # results = model.inference_on_dataset(questions, batch_size=args.batch_size)
    # print(args.batch_size)
    results = model.inference_on_dataset(repeated_questions, batch_size=args.batch_size)
    sampled_responses = []
    for i in range(0, len(questions)):
        sampled_responses.append([])
        for j in range(0, args.sample_num):
            sampled_responses[i].append(results[j*len(questions)+i])
    output = []
    for i, question in enumerate(questions):
        output.append({
            "question": question,
            "sampled_responses": sampled_responses[i],
            "question_index": i
        })
    
    with open(args.output_path, "w") as f:
        json.dump(output, f)


class VanillaLLM:
    def __init__(self, model_path, generation_config=None, prompt_dir="prompt_templates/", prompt_key="zero_shot", device="auto"):
        self.model = BaseGenerator(model_path=model_path, generation_config=generation_config, device=device)
        self.prompt_loader = PromptTemplateLoader(template_dir_path=prompt_dir)
        self.prompt_key = prompt_key

    def inference_on_dataset(self, questions, batch_size=16):
        prompts = [self.prompt_loader.construct_prompt(self.prompt_key, {"question": question}) for question in questions]
        prompts = [self.prompt_loader.construct_chat_input(prompt, tokenizer=self.model.base_tokenizer) for prompt in prompts]
        responses = self.model.inference_on_data(prompts, batch_size=batch_size)
        return responses


class DoLaInference:
    def __init__(self, model_path, generation_config=None, prompt_dir="prompt_templates/", prompt_key="zero_shot", device="auto"):
        if generation_config is None:
            generation_config = {}
        generation_config["do_layers"] = "high"
        if "generation_penalty" not in generation_config.keys():
            generation_config["generation_penalty"] = 1.2
        self.model = BaseGenerator(model_path=model_path, generation_config=generation_config, device=device)
        self.prompt_loader = PromptTemplateLoader(template_dir_path=prompt_dir)
        self.prompt_key = prompt_key

    def inference_on_dataset(self, questions, batch_size=16):
        prompts = [self.prompt_loader.construct_prompt(self.prompt_key, {"question": question}) for question in questions]
        prompts = [self.prompt_loader.construct_chat_input(prompt, tokenizer=self.model.base_tokenizer) for prompt in prompts]
        responses = self.model.inference_on_data(prompts, batch_size=batch_size)
        return responses


class USCInference:
    def __init__(self, voting_num, model_path, generation_config=None, prompt_dir="prompt_templates/", prompt_key="zero_shot", device="auto"):
        self.voting_num = voting_num
        self.model = BaseGenerator(model_path=model_path, generation_config=generation_config, device=device)
        self.prompt_loader = PromptTemplateLoader(template_dir_path=prompt_dir)
        self.prompt_key = prompt_key

    def inference_on_dataset(self, questions, batch_size, sampled_responses=None):
        ori_prompts = [self.prompt_loader.construct_prompt(self.prompt_key, {"question": question}) for question in questions]
        if sampled_responses is None:
            prompts = [self.prompt_loader.construct_chat_input(p, tokenizer=self.model.base_tokenizer) for p in ori_prompts]
            inputs = []
            for p in prompts:
                for _ in range(self.voting_num):
                    inputs.append(copy.deepcopy(p))   
            responses = self.model.inference_on_data(inputs, batch_size=batch_size)  
            sampled_responses = [responses[i:i+self.voting_num] for i in range(0, len(responses), self.voting_num)]
            
        if self.voting_num == 1:
            return [rs[0] for rs in sampled_responses]

        sampled_responses_text = []
        for one_set_responses in sampled_responses:
            sampled_responses_text.append('\n\n'.join(f"Reponses {i}: {r}" for i, r in enumerate(one_set_responses)))
        prompts = [self.prompt_loader.construct_prompt("usc_template", {"question": questions[i], "sampled_responses": sampled_responses_text[i]}) for i in range(len(questions))]
        prompts = [self.prompt_loader.construct_chat_input(prompt, tokenizer=self.model.base_tokenizer) for prompt in prompts]
        responses_num_result = self.model.inference_on_data(prompts, batch_size=batch_size)
        # for i, r in enumerate(responses_num_result):
        #     print(r)
        #     print('------')
        responses = []
        for i, one_num_response in enumerate(responses_num_result):
            find_num = False
            one_num_response = one_num_response.lower()
            # print(one_num_response)
            while one_num_response.find("response ") != -1:
                one_num_response = one_num_response[one_num_response.find("response ")+len("response "):]
                # extract the number at the beginning
                if one_num_response[0].isdigit():
                    num = 0
                    while one_num_response[0].isdigit():
                        num = num * 10 + int(one_num_response[0])
                        one_num_response = one_num_response[1:]
                        if len(one_num_response) == 0:
                            break
                    if num < len(sampled_responses[i]):
                        responses.append(sampled_responses[i][num])
                        find_num = True
                        break
            if not find_num:
                print(f"Error: {one_num_response}")
                responses.append(sampled_responses[i][0])
        return responses


class SelfReflectionInference:
    def __init__(self, voting_num, model_path, generation_config=None, prompt_dir="prompt_templates/", prompt_key="zero_shot", device="auto"):
        self.voting_num = voting_num
        self.model = BaseGenerator(model_path=model_path, generation_config=generation_config, device=device)
        self.prompt_loader = PromptTemplateLoader(template_dir_path=prompt_dir)
        self.prompt_key = prompt_key

    def inference_on_dataset(self, questions, batch_size, sampled_responses=None):
        ori_prompts = [self.prompt_loader.construct_prompt(self.prompt_key, {"question": question}) for question in questions]
        if sampled_responses is None:
            prompts = [self.prompt_loader.construct_chat_input(p, tokenizer=self.model.base_tokenizer) for p in ori_prompts]
            inputs = []
            for p in prompts:
                for _ in range(self.voting_num):
                    inputs.append(copy.deepcopy(p))   
            responses = self.model.inference_on_data(inputs, batch_size=batch_size)  
            sampled_responses = [responses[i:i+self.voting_num] for i in range(0, len(responses), self.voting_num)]
        
        sampled_responses_text = []
        for one_set_responses in sampled_responses:
            sampled_responses_text.append('\n\n'.join(f"Reponses {i}: {r}" for i, r in enumerate(one_set_responses)))
        prompts = [self.prompt_loader.construct_prompt("self_reflection_template", \
                    {"question": questions[i], "sampled_responses": sampled_responses_text[i]})+questions[i] for i in range(len(questions))]
        # print(prompts[-1])
        prompts = [self.prompt_loader.construct_chat_input(prompt, tokenizer=self.model.base_tokenizer) for prompt in prompts]
        responses = self.model.inference_on_data(prompts, batch_size=batch_size)
        for i, r in enumerate(responses):
            key_sentence = "The answer to this question should be:"
            if r.find(key_sentence) != -1:
                responses[i] = r[r.find(key_sentence)+len(key_sentence):]
        return responses


class VotingDecodingInference:
    def __init__(self, 
                voting_num, 
                model_path, 
                generation_config=None, 
                prompt_dir="prompt_templates/", 
                standard_prompt_key="zero_shot",
                voting_decoding_prompt_key="voting_decoding_template",
                device="auto"):
        self.voting_num = voting_num
        self.model_path = model_path
        self.generation_config = generation_config
        self.prompt_loader = PromptTemplateLoader(template_dir_path=prompt_dir)
        self.prompt_key = standard_prompt_key
        self.voting_decoding_prompt_key = voting_decoding_prompt_key
        self.device=device
    
    def inference_on_dataset(self, questions, batch_size=32, sampled_responses=None, max_length=2048):
        ori_prompts = [self.prompt_loader.construct_prompt(self.prompt_key, {"question": question}) for question in questions]
        if sampled_responses is None:
            self.model = BaseGenerator(model_path=self.model_path, generation_config=self.generation_config)
            prompts = [self.prompt_loader.construct_chat_input(p, tokenizer=self.model.base_tokenizer) for p in ori_prompts]
            inputs = []
            for p in prompts:
                for _ in range(self.voting_num):
                    inputs.append(copy.deepcopy(p))   
            responses = self.model.inference_on_data(inputs, batch_size=batch_size)  
            sampled_responses = [responses[i:i+self.voting_num] for i in range(0, len(responses), self.voting_num)]

        middle_results = copy.deepcopy(sampled_responses)
        n_sample_per_batch = batch_size // self.voting_num
        batch_size = n_sample_per_batch * self.voting_num
        generator = GeneratorWithVD(voting_num=self.voting_num, model_path=self.model_path, device=self.device)
        final_responses = []
        i=-1
        new_inputs = []
        for one_output in tqdm(sampled_responses):
            i+=1
            original_prompt = ori_prompts[i]
            for j, sampled_response in enumerate(one_output):
                new_input = self.prompt_loader.construct_prompt("voting_decoding_template", {"question": questions[i], "sampled_response": sampled_response})
                new_inputs.append(self.prompt_loader.construct_chat_input(new_input, tokenizer=generator.base_tokenizer))
            # print(new_inputs)
            if len(new_inputs) == n_sample_per_batch * self.voting_num:
                final_responses += generator.inference_n_samples(new_inputs,
                                                                do_sample=True,
                                                                temperature=0.01,
                                                                max_length=max_length)
                new_inputs = []
        if len(new_inputs) > 0:
            final_responses += generator.inference_n_samples(new_inputs,
                                                            do_sample=True,
                                                            temperature=0.01)
        for i, r in enumerate(final_responses):
            key_sentence = "The answer to this question should be: "
            if r.find(key_sentence) != -1:
                final_responses[i] = r[r.find(key_sentence)+len(key_sentence):]
        return_dict = []
        for i, one_response in enumerate(final_responses):
            return_dict.append({"input": ori_prompts[i],
                                "middle_results": middle_results[i],
                                "output": one_response})
        return final_responses, return_dict