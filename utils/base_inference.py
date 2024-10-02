import sys
import os
sys.path.append(os.path.dirname(__file__))
import json
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModel
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from voting_decoding import LlamaForCausalLMWithVD, MistralForCausalLMWithVD, Qwen2ForCausalLMWithVD, GemmaForCausalLMWithVD
import copy
import json
# from PromptTemplateLoader import *

# prompt_template_loader = PromptTemplateLoader()
base_model_prompt = """
Give a bullet point biography of Aaron Sloman highlighting their contributions and achievements as a computer scientist.
Each fact separated with a new line character. The biography should include 5 facts."""

base_model_prompt2 = """Suppose you are an INTP. Which one do you prefer?\nA. Gentle\nB. Strong
My answer is"""

LLAMA_CKPT_DIR = "meta-llama/Meta-Llama-3-8B-Instruct"
# LLAMA_CKPT_DIR = "meta-llama/Llama-2-7b-chat-hf"
MISTRAL_CKPT_DIR = "mistralai/Mistral-7B-Instruct-v0.2"
GEMMA_CKPT_DIR = "google/gemma-7b-it"
Llama3_1_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

class BaseGenerator:
    def __init__(self, device="cuda:0", generation_config=None, model_path=LLAMA_CKPT_DIR):
        # print(os.environ['CUDA_VISIBLE_DEVICES'])
        self.base_model_path = model_path
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
        # print(self.base_model)
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, device=device, padding_side="left", trust_remote_code=True)
        # self.base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_tokenizer.padding_side = 'left'
        self.base_model.eval()
        if device == "auto":
            self.device = 0
        else:
            self.device = int(device[5:])
        print(f"self.device: {self.device}")
        # self.base_model.to(device)
        if generation_config is None:
            generation_config = {}
        if "max_length" not in generation_config.keys():
            generation_config["max_length"] = 8192
        generation_config["pad_token_id"] = self.base_tokenizer.pad_token_id
        if model_path.find('glm-4') != -1:
            generation_config["eos_token"] = "<|user|>"
            generation_config["eos_token_id"] = 151336
        else:
            generation_config["eos_token"] = self.base_tokenizer.eos_token
            generation_config["eos_token_id"] = self.base_tokenizer.eos_token_id
        self.generation_config = GenerationConfig(**generation_config)
        print("Generation config: ", self.generation_config)


    def inference_on_data(self,
                          prompts,
                          batch_size=16,
                          max_length=8192,
                          tqdm_visible=True):
        responses = []
        step = 0
        if tqdm_visible:
            for i in tqdm(range(0, len(prompts), batch_size)):
                step += 1
                batch = prompts[i:i+batch_size]
                batch_responses = self.inference_on_one_batch(batch)
                responses.extend(batch_responses)
        else:
            for i in range(0, len(prompts), batch_size):
                step += 1
                batch = prompts[i:i+batch_size]
                batch_responses = self.inference_on_one_batch(batch)
                responses.extend(batch_responses)
        return responses

    def inference_on_one_batch(self, prompts):
        base_inputs = self.base_tokenizer(prompts, return_tensors="pt", padding=True)
        base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}

        # calculate the time for generation
        outputs = self.base_model.generate(
            **base_inputs,
            generation_config=self.generation_config)

        # print(self.base_tokenizer.eos_token_id)
        responses = self.base_tokenizer.batch_decode(outputs[:, base_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return responses

    def inference_one_sample(self, prompt):
        base_inputs = self.base_tokenizer([prompt], return_tensors="pt", padding=True)
        base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}
        outputs = self.base_model.generate(
            **base_inputs,
            generation_config=self.generation_config)

        # print(self.generation_config.eos_token)
        responses = self.base_tokenizer.batch_decode(outputs[:, base_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return responses[0]


class GeneratorWithVD:
    def __init__(self, device="cuda:0", model_path=LLAMA_CKPT_DIR, merge_strategy="vote_for_best", voting_num=1):
        self.voting_num = voting_num
        self.base_model_path = model_path
        if "llama" in model_path.lower():
            self.base_model = LlamaForCausalLMWithVD.from_pretrained(self.base_model_path, torch_dtype=torch.float16,
                                                                     device_map=device)
        elif "qwen" in model_path.lower():
            self.base_model = Qwen2ForCausalLMWithVD.from_pretrained(self.base_model_path, torch_dtype=torch.float16,
                                                                     device_map=device)
        elif "mistral" in model_path.lower():
            self.base_model = MistralForCausalLMWithVD.from_pretrained(self.base_model_path, torch_dtype=torch.float16,
                                                                       device_map=device)
        elif "gemma" in model_path.lower():
            self.base_model = GemmaForCausalLMWithVD.from_pretrained(self.base_model_path, torch_dtype=torch.float16,
                                                                     device_map=device)
        elif "glm" in model_path.lower():
            self.base_model = AutoModel.from_pretrained(self.base_model_path, torch_dtype=torch.float16,
                                                                     device_map=device, offload_folder="offload", trust_remote_code=True)
        self.base_model.voting_num = voting_num
        self.base_model.merge_strategy = merge_strategy
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, device=device, trust_remote_code=True)
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        if model_path.find('glm-4') != -1:
            self.base_tokenizer.eos_token = "<|user|>"
            self.base_tokenizer.eos_token_id = 151336
        # self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_tokenizer.padding_side = 'left'
        self.base_model.eval()
        if device == "auto":
            self.device = 0
        else:
            self.device = int(device[5:])
        # self.base_model.to(device)

    def inference_n_samples(self,
                              prompt_list,
                              do_sample=True,
                              repetition_penalty=1.0,
                              temperature=0.7,
                              top_p=0.8,
                              max_length=8192,
                              max_new_tokens=2048):
        base_inputs = self.base_tokenizer(prompt_list, return_tensors="pt", padding=True)
        base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}

        generation_cfg = GenerationConfig(
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.base_tokenizer.pad_token_id,
            repetition_penalty = repetition_penalty,
            eos_token = self.base_tokenizer.eos_token,
            eos_token_id=self.base_tokenizer.eos_token_id
        )
        outputs = self.base_model.generate(
            **base_inputs,
            generation_config=generation_cfg)

        responses = self.base_tokenizer.batch_decode(outputs[:, base_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        responses = [responses[i] for i in range(0, len(responses), self.voting_num)]
        return responses


if __name__ == "__main__":
    model = GeneratorWithVD(voting_num=2)
    prompt_list = [base_model_prompt, base_model_prompt2]
    result = model.inference_n_samples(prompt_list, max_new_tokens=16)
    print(result[0])