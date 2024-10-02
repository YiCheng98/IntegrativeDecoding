import sys
sys.path.append("../")

import nltk
# import torch
from nltk import tokenize
from vllm import LLM, SamplingParams
import string
import numpy as np
import spacy
import rank_bm25
import os
import json
import re
import itertools
from utils.base_inference import *
import copy
import argparse
from utils.my_API_call import *

from openai import OpenAI, AsyncOpenAI
import asyncio
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

async_local_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
model = "llama3.1"

parser = argparse.ArgumentParser()
parser.add_argument("--divide_all_files", action="store_true")
parser.add_argument("--key_word", type=str, default=None)
parser.add_argument("--eval_dir", type=str, default="output/")
parser.add_argument("--file_name", type=str, default="mistral_vanilla_llm_test.json")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--cover_old_results", action="store_true")
# parser.add_argument("--output_dir", type=str, default="output/qwen_voting_decoding_voting_16_test.json")

args = parser.parse_args()
print(args)
args.output_dir = os.path.join(args.eval_dir, "atomic_facts")
available_gpus = 4

# torch.multiprocessing.set_start_method('spawn')

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

MONTHS = [
    m.lower()
    for m in [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    ]
]
SPACY_MODEL = spacy.load('en_core_web_sm')
DEMON_DIR = 'demons/'
_SENTENCE = 'sentence'
_ATOMIC_FACTS = 'atomic_facts'
ATOMIC_FACT_INSTRUCTION = """\
Instructions:
1. Start by examining the provided sentence to identify information about a specific entity.
2. Define an "atomic fact" as a sentence that relays only one piece of information regarding the entity in question.
3. Decompose the sentence into a list of atomic facts, ensuring that each fact presents a unique piece of information without repeating details found in other atomic facts.
4. Refer to previous examples as a guide for accurately performing this task.
5. Format your output as a bulleted list where each atomic fact begins with "- ". Exclude any additional formatting.
"""

# The following are some expamples of sentences and their corresponding atomic facts.


def text_to_sentences(text: str, separator: str = '- ') -> list[str]:
    """Transform InstructGPT output into sentences."""
    sentences = text.split(separator)[1:]
    sentences = [
        sentence[:sentence.find('\n')] if '\n' in sentence else sentence
        for sentence in sentences
    ]
    sentences = [
        sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip()
        for sent in sentences if len(sent)>1
    ]

    if sentences:
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.'
    else:
        sentences = []

    return sentences


def extract_numeric_values(text):
  pattern = r'\b\d+\b'  # regular expression pattern for integers
  numeric_values = re.findall(
      pattern, text
  )  # find all numeric values in the text
  return set(
      [value for value in numeric_values]
  )  # convert the values to float and return as a list


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_num(text):
  try:
    _ = int(text)
    return True
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def is_date(text):
#   print(text)
  text = normalize_answer(text)
#   print(text)
  for token in text.split(' '):
    if (not is_num(token)) and token not in MONTHS:
      return False

  return True



def detect_entities(text, nlp):
  """Detect entities from the text."""
  doc, entities = nlp(text), set()
#   print("--------------")
#   print(f"doc: {doc}")
  def _add_to_entities(text):
    if '-' in text:
      for each in text.split('-'):
        entities.add(each.strip())
    else:
      entities.add(text)
 
#   print(f"entities: {doc.ents}")
  for ent in doc.ents:
    # print(f"ent: {ent}")
    # spacy often has errors with other types of entities
    if ent.label_ in [
        'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
    ]:
      if is_date(ent.text):
        _add_to_entities(ent.text)
      else:
        for token in ent.text.split():
          if is_date(token):
            _add_to_entities(token)

  for new_ent in extract_numeric_values(text):
    if not np.any([new_ent in ent for ent in entities]):
      entities.add(new_ent)

  return entities


def postprocess_atomic_facts(in_atomic_facts, para_breaks, nlp):
  """Postprocess atomic facts."""
  verbs = [
      'born.',
      ' appointed.',
      ' characterized.',
      ' described.',
      ' known.',
      ' member.',
      ' advocate.',
      'served.',
      'elected.',
  ]
  permitted_verbs = ['founding member.']
  atomic_facts, new_atomic_facts, new_para_breaks = [], [], []

  for i, (sent, facts) in enumerate(in_atomic_facts):
    sent = sent.strip()

    if len(sent.split()) == 1 and i not in para_breaks and i > 0:
      assert i not in para_breaks
      atomic_facts[-1][0] += ' ' + sent
      atomic_facts[-1][1] += facts
    else:
      if i in para_breaks:
        new_para_breaks.append(len(atomic_facts))

      atomic_facts.append([sent, facts])

  for _, (sent, facts) in enumerate(atomic_facts):
    entities = detect_entities(sent, nlp)
    covered_entities, new_facts = set(), []

    for i, fact in enumerate(facts):
      if any([fact.endswith(verb) for verb in verbs]) and not any(
          [fact.endswith(verb) for verb in permitted_verbs]
      ):
        if any([
            fact[:-1] in other_fact
            for j, other_fact in enumerate(facts)
            if j != i
        ]):
          continue

      sent_entities = detect_entities(fact, nlp)
      covered_entities |= set([e for e in sent_entities if e in entities])
      new_entities = sent_entities - entities

      if new_entities:
        do_pass = False

        for new_ent in new_entities:
          pre_ent = None

          for ent in entities:
            if ent.startswith(new_ent):
              pre_ent = ent
              break

          if pre_ent is None:
            do_pass = True
            break

          fact = fact.replace(new_ent, pre_ent)
          covered_entities.add(pre_ent)

        if do_pass:
          continue

      if fact in new_facts:
        continue

      new_facts.append(fact)

    # there is a bug in spacy entity linker, so just go with the previous facts
    try:
      assert entities == covered_entities
    except AssertionError:
      new_facts = facts

    new_atomic_facts.append((sent, new_facts))

  return new_atomic_facts, new_para_breaks


def best_demos(query, bm25, demons_sents, k):
  tokenized_query = query.split(' ')
  top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
  return top_machings


def open_file_wrapped(filepath: str, **kwargs):
    try:
        return open(filepath, **kwargs)
    except:
        return open("../../" + filepath, **kwargs)


def detect_initials(text):
    pattern = r'[A-Z]\. ?[A-Z]\.'
    match = re.findall(pattern, text)
    return [m for m in match]


def fix_sentence_splitter(curr_sentences, initials):
    """Fix sentence splitter issues."""
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split('.') if t.strip()]

            for i, (sent1, sent2) in enumerate(
                zip(curr_sentences, curr_sentences[1:])
            ):
                if sent1.endswith(alpha1 + '.') and sent2.startswith(alpha2 + '.'):
                    # merge sentence i and i+1
                    curr_sentences = (
                        curr_sentences[:i]
                        + [curr_sentences[i] + ' ' + curr_sentences[i + 1]]
                        + curr_sentences[i + 2 :]
                    )
                    break
    sentences, combine_with_previous = [], None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += ' ' + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


class AtomicFactGenerator(object):
    """Atomic fact generator."""

    def __init__(self, demon_dir=DEMON_DIR, device=0):
        self.nlp = SPACY_MODEL
        self.is_bio = True
        sample_params = SamplingParams(max_tokens=2048)#, stop=["<end>"])
        self.demon_path = os.path.join(demon_dir, 'demons.json' if self.is_bio else 'demons_complex.json')
        # get the demos
        with open_file_wrapped(self.demon_path, mode='r') as f:
            self.demons = json.load(f)
        tokenized_corpus = [doc.split(' ') for doc in self.demons.keys()]
        self.bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
        LLAMA3_1_70B_chat = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        with open("prompt_templates/divide_atomic_facts.txt", 'r') as f:
            self.divide_prompt_template = f.read()
        # self.llm = LLM(LLAMA3_1_70B_chat, tensor_parallel_size=4, swap_space=16)
        # self.llm = BaseGenerator(device="cuda:0", model_path=Llama3_1_PATH, generation_config={"max_length": 1200})
        # self.llm = BaseGenerator(device=f"cuda:{device}", model_path=LLAMA_CKPT_DIR, generation_config={"max_length": 2048})
        # self.llm = BaseGenerator(device="cuda:0", model_path=MISTRAL_PATH)


    def run(self, generation: str):
        """Convert the generation into a set of atomic facts."""
        assert isinstance(generation, str), 'generation must be a string'
        paragraphs = [
            para.strip() for para in generation.split('\n') if para.strip()
        ]
        return self.get_atomic_facts_from_paragraph(
            paragraphs
        )


    def get_atomic_facts_from_paragraph(self, paragraphs):
        """Get the atomic facts from the paragraphs."""
        sentences, para_breaks = [], []

        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(len(sentences))

            initials = detect_initials(paragraph)
            curr_sentences = tokenize.sent_tokenize(paragraph)
            curr_sentences_2 = tokenize.sent_tokenize(paragraph)
            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)
            # ensure the crediability of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (
                paragraph,
                curr_sentences,
                curr_sentences_2,
            )
            sentences += curr_sentences

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence(
            [
                sent
                for i, sent in enumerate(sentences)
                if not (
                    not self.is_bio
                    and (
                        (
                            #i == 0
                            #and (
                                sent.startswith('Sure')
                                or sent.startswith('Here are')
                            #)
                        )
                        or (
                            #i == len(sentences) - 1
                            #and (
                                sent.startswith('Please')
                                or sent.startswith('I hope')
                                or sent.startswith('Here are')
                            #)
                        )
                    )
                )
            ]
        )

        atoms = atoms_or_estimate

        atomic_facts_pairs = []

        for i, sent in enumerate(sentences):
            if not self.is_bio and (
                (i == 0 and (sent.startswith('Sure') or sent.startswith('Here are')))
                or (
                    i == len(sentences) - 1
                    and (
                        sent.startswith('Please')
                        or sent.startswith('I hope')
                        or sent.startswith('Here are')
                    )
                )
            ):
                atomic_facts_pairs.append((sent, []))
            elif self.is_bio and sent.startswith(
                'This sentence does not contain any facts'
            ):
                atomic_facts_pairs.append((sent, []))
            elif (
                sent.startswith('Sure')
                or sent.startswith('Please')
                or (i == 0 and sent.startswith('Here are'))
            ):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(
                atomic_facts_pairs, list(para_breaks), self.nlp
            )

        return atomic_facts_pairs, para_breaks


    def get_init_atomic_facts_from_sentence(self, sentences):
        """Get the initial atomic facts from the sentences."""
        is_bio, demons = self.is_bio, self.demons
        prompts, prompt_to_sent, atoms = [], {}, {}
        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        for sentence in sentences:
            # if sentence in atoms:
            #     continue

            # top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            # prompt = ''

            # for i in range(n):
            #     prompt += (
            #         # 'Please breakdown the following sentence into independent facts:'
            #         'Sentence:'
            #         ' {}\n'.format(list(demons.keys())[i])
            #     )
            #     prompt += "Atomic Facts:\n"
            #     for fact in demons[list(demons.keys())[i]]:
            #         prompt += '- {}\n'.format(fact)
            #     prompt += '\n'

            # for match in top_machings:
            #     prompt += (
            #         # 'Please breakdown the following sentence into independent facts:'
            #         'Sentence:'
            #         ' {}\n'.format(match)
            #     )

            #     for fact in demons[match]:
            #         prompt += '- {}\n'.format(fact)

            #     prompt += '\n'

            # # Add eval example
            # # prompt += ATOMIC_FACT_INSTRUCTION
            # prompt += (
            #     'Now list the atomic facts for the following sentence. \nSentence:'
            #     ' {}\n'.format(sentence)
            # )
            # prompt += "Atomic Facts:"

            # prompts.append(prompt)
            prompt = self.divide_prompt_template.replace("{sentence}", sentence)
            prompts.append(copy.deepcopy(prompt))
            prompt_to_sent[prompt] = sentence

        messages = [[{"role": "system", "content": "You are a helpful assistant. Strictly follow the given instruction to generate a response."},{"role": "user", "content": prompt}] for prompt in prompts]
        # self.llm.base_tokenizer.chat_template = Llama3_1_CHAT_TEMPLATE
        # messages = [self.llm.base_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in messages]
        # print(messages[-1])
        # print(json.dumps(messages, indent=4))
        # print(len(messages))
        # print("Extracting atomic facts...")
        outputs=asyncio.run(call_LLM_async(messages, client=async_local_client, model="llama3.1", batch_size=16))
        for i, o in enumerate(outputs):
            pos = o.find("The refined list should be:")
            if pos != -1:
                outputs[i] = o[pos+len("The refined list should be:"):]
        # for i in range(len(outputs)):
        #     print(f"Sentece: {sentences[i]}")
        #     print(outputs[i])

        # for i in range(len(sentences)):
        #     print(sentences[i])
        #     print(outputs[i])
        #     print("=="*20)
        # outputs = self.llm.inference_on_data(messages, batch_size=16, tqdm_visible=False)
        # print(json.dumps(outputs, indent=4))
        # outputs = asyncio.run(
        #     my_api_call_async(messages, model="gpt-4o-free", batch_size=16, tqdm_visible=True))
        for i, output in enumerate(outputs):
            sentences_from_output = text_to_sentences(output)
            if not sentences_from_output:  # account for markdown lists
                sentences_from_output = text_to_sentences(output, separator='* ')
            prompt = prompts[i]
            atoms[prompt_to_sent[prompt]] = sentences_from_output

        for key, value in demons.items():
            if key not in atoms:
                atoms[key] = value
        return atoms

def convert_atomic_facts_to_dicts(
    outputted_facts: list[tuple[str, list[str]]]
):
  return [
      {_SENTENCE: sentence, _ATOMIC_FACTS: identified_atomic_facts}
      for sentence, identified_atomic_facts in outputted_facts
  ]

def divide_one_file(fact_divider: AtomicFactGenerator, file_path: str, output_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for sample in tqdm(data, desc="Annotation Progress"):
        generated_answer = sample["generated_answer"]
        key_sentence = "The answer to this question should be:"
        if generated_answer.find(key_sentence) != -1:
            generated_answer = generated_answer[generated_answer.find(key_sentence)+len(key_sentence):]
            # print(generated_answer)
        extracted_fact_info = fact_divider.run(generated_answer)
        facts, _ = extracted_fact_info
        facts_as_dict = convert_atomic_facts_to_dicts(facts)
        all_atomic_facts = list(
            itertools.chain.from_iterable([f[_ATOMIC_FACTS] for f in facts_as_dict])
        )
        extracted_fact_info = {
            'num_claims': len(all_atomic_facts),
            'sentences_and_atomic_facts': facts,
            'all_atomic_facts': facts_as_dict,
        }
        # for item in facts:
        #     print(item[0])
        #     print("\n- ".join(item[1]))
        #     print()
        sample["extracted_fact_info"] = copy.deepcopy(extracted_fact_info)
        # print(extracted_fact_info["num_claims"])
    with open(output_path, 'w') as f:
        f.write(json.dumps(data))
    print(f"Finish annotating {output_path}")
    

if __name__=="__main__":
    fact_divider = AtomicFactGenerator(device=args.device)
    if args.divide_all_files:
        file_names = os.listdir(args.eval_dir)
        if args.key_word:
            file_names = [f for f in file_names if f.find(args.key_word)!=-1]# or f.find("vanilla_llm")!=-1]
        for file_name in tqdm(file_names, desc="File progress: "):
            file_path = os.path.join(args.eval_dir, file_name)
            output_path = os.path.join(args.output_dir, file_name)
            if os.path.exists(output_path) and not args.cover_old_results:
                print(f"Skipping {file_name}")
                continue
            else:
                print(f"Dividing {file_name}")
            try:
                divide_one_file(fact_divider, file_path, output_path)
            except:
                print(f"Fail to annotate {file_name}.")
    else:
        # fact_divider.run("In 2010, Cedric Villani was awarded the prestigious Fields Medal, often considered the Nobel Prize of mathematics, for his significant contributions to the theory of partial differential equations and his work on the Boltzmann equation.")
        file_path = os.path.join(args.eval_dir, args.file_name)
        output_path = os.path.join(args.output_dir, args.file_name)
        divide_one_file(fact_divider, file_path, output_path)
        