import os

from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, stop_after_attempt, wait_random_exponential
import asyncio
import nest_asyncio
from tqdm import tqdm

# supported_model_map = {
#     "gpt-3.5": "chatgpt",
#     "gpt-3.5-turbo": "chatgpt",
#     "gpt-4": "gpt-4-32k",
#     "gpt-4-turbo": "gpt-4-32k"
# }


def construct_client(asy=False):
    if not asy:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    else:
        client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return client




client = construct_client()
asy_client = construct_client(asy=True)


def call_LLM(messages, model="gpt-4", generation_config=None):
    # if model in supported_model_map.keys():
    #     model = supported_model_map[model]
    if generation_config is None:
        result = client.chat.completions.create(model=model,
                                                messages=messages,
                                                stream=False)
    else:
        result = client.chat.completions.create(model=model,
                                                messages=messages,
                                                stream=False,
                                                temperature=generation_config["temperature"],
                                                top_p=generation_config["top_p"])
    response = result.choices[0].message.content
    return response


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
async def async_fetch_response(messages,
                               model="chatgpt",
                               generation_config=None):
    # if model in supported_model_map.keys():
    #     model = supported_model_map[model]
    client = asy_client
    if generation_config is None:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=generation_config["temperature"],
        )
    content = response.choices[0].message.content
    return content


async def call_LLM_async(messages_list,
                         model="chatgpt",
                         generation_config=None,
                         batch_size=16,
                         tqdm_visible=False):
    nest_asyncio.apply()
    tasks = [async_fetch_response(messages, model=model, generation_config=generation_config) for messages in
             messages_list]
    task_batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    responses = []
    # print(f"Number of batches: {len(task_batches)}")
    if tqdm_visible:
        for task_batch in tqdm(task_batches):
            responses.extend(await asyncio.gather(*task_batch))
    else:
        for task_batch in task_batches:
            responses.extend(await asyncio.gather(*task_batch))
    # responses = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
    return responses


if __name__ == "__main__":
    model = "chatgpt"
    messages = [{"role": "user", "content": "Hello, world!"}]
    messages_list = [messages] * 4
    # call_LLM(messages, model=model)
    responses = asyncio.run(call_LLM_async(messages_list, model=model))
    print(responses)
