import json
import os
import random

cur_dir = os.path.dirname(__file__)

def load_longfact(type="objects", max_sample_num=120):
    data_file = os.path.join(cur_dir, f"dataset/longfact_{type}_random.json")
    dataset = json.load(open(data_file, "r"))
    return dataset[:max_sample_num]

if __name__ == "__main__":
    load_longfact()