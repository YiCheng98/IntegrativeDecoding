import copy
import os


DEFAULT_SYSTEM_PROMPT = '''
You are a helpful assistant. Strictly follow the given instruction to generate a response.
'''
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.

LLAMA3_INSTRUCT_INPUT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

LLAMA2_CHAT_INPUT_TEMPLATE = "<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_message} [/INST]"

MISTRAL_INSTRUCT_INPUT_TEMPLATE = "<s>[INST]{user_message}[/INST]"

class PromptTemplateLoader:
    def __init__(self, template_dir_path="DEFAULT_TEMPLATE_DIR_PATH"):
        if template_dir_path == "DEFAULT_TEMPLATE_DIR_PATH":
            self.template_dir_path = os.path.join(os.path.dirname(__file__), "prompt_templates")
        else:
            self.template_dir_path = template_dir_path
        self.prompt_templates = {}
        self.system_instruction_templates = {}

    def load_prompt_template(self, prompt_type):
        if prompt_type not in self.prompt_templates:
            ## Load the prompt template from the file. If the file does not exist, raise an error.
            template_file_path = os.path.join(self.template_dir_path, f"{prompt_type}.txt")
            if not os.path.exists(template_file_path):
                raise FileNotFoundError(f"Prompt template file for prompt type '{prompt_type}' does not exist.")
            with open(template_file_path, "r", encoding="utf-8") as f:
                self.prompt_templates[prompt_type] = f.read()
        return self.prompt_templates[prompt_type]

    def load_system_instruction_template(self, prompt_type):
        if prompt_type not in self.system_instruction_templates:
            ## Load the prompt template from the file. If the file does not exist, raise an error.
            template_file_path = os.path.join(self.template_dir_path, f"{prompt_type}_system_instruction.txt")
            if not os.path.exists(template_file_path):
                raise FileNotFoundError(f"System instruction template file for prompt type '{prompt_type}' does not exist.")
            with open(template_file_path, "r", encoding="utf-8") as f:
                self.system_instruction_templates[prompt_type] = f.read()
        return self.system_instruction_templates[prompt_type]

    def my_format(self, prompt_template, template_placeholders):
        prompt = copy.deepcopy(prompt_template)
        for placeholder, value in template_placeholders.items():
            prompt = prompt.replace('{'+placeholder+'}', value)
        return prompt
    def construct_prompt(self, prompt_key, template_placeholders):
        prompt_template = self.load_prompt_template(prompt_key)
        # prompt = self.my_format(prompt_template, template_placeholders)
        # try formatting the prompt template with the placeholders. If error, use my_format
        try:
            prompt = prompt_template.format(**template_placeholders)
        except KeyError as e:
            prompt = self.my_format(prompt_template, template_placeholders)
        return prompt

    def construct_system_instruction(self, prompt_key, template_placeholders):
        system_instruction_template = self.load_system_instruction_template(prompt_key)
        try:
            system_instructions = system_instruction_template.format(**template_placeholders)
        except KeyError as e:
            system_instructions = self.my_format(system_instruction_template, template_placeholders)
        return system_instructions
    
    def construct_chat_input(self, user_message, tokenizer):
        chat = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}]
        if hasattr(tokenizer, "chat_template"):
            chat_template = tokenizer.chat_template
            if chat_template.find("System role not supported"):
                chat = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt