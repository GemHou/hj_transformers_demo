from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
import time

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
# from hj_inference_textGeneration_tokenizer_CausalLM import prepare_task

block_size = 128


def prepare_task():
    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family

    (except for Alexei and Maria) are discovered.

    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the

    remainder of the story. 1883 Western Siberia,

    a young Grigori Rasputin is asked by his father and a group of men to perform magic.

    Rasputin has a vision and denounces one of the men as a horse thief. Although his

    father initially slaps him for making such an accusation, Rasputin watches as the

    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of

    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,

    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    PADDING_TEXT = """ """

    prompt = "Today the weather is really nice and I am planning on "

    full_prompt = "VILPNNDRHQITDTTNGHYAPVTYIQVEAPTGTFIASGVVVGKDTLLTNKHVVDATHGDPHALKAFPSAINQDNYPNGGFTAEQITKYSGEGDLAIVKFSPNEQNKHIGEVVKPATMSNNAETQTNQNITVTGYPGDKPVATMWESKGKITYLKGEAMQYDLSTTGGNSGSPVFNEKNEVIGIHWGGVPNEFNGAVFINENVRNFLKQNIEDINFANDDQPNNPDNPDNPNNPDNPNNPDNPNNPDEPNNPDNPNNPDNPDNGDNNNSDNPDAA"

    prompt = full_prompt[:int(len(full_prompt)*0.4)]  # int(len(full_prompt)*0.3)

    return PADDING_TEXT, prompt


def main():
    model_name = "distilgpt2"  # bert-base-cased-finetuned-mrpc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)  # bert-base-cased
    model.load_state_dict(torch.load("../para_temp.pt"))

    PADDING_TEXT, prompt = prepare_task()

    inputs = tokenizer(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"]  # PADDING_TEXT +

    prompt_length = len(tokenizer.decode(inputs[0]))

    print("forward... waiting...")
    start_time = time.time()
    outputs = model.generate(inputs, max_length=prompt_length-46+256, do_sample=True, top_p=0.95, top_k=60)  # max_length=250
    print("forward time: ", time.time()-start_time)
    outputs = outputs[0]
    outputs = tokenizer.decode(outputs)

    generated = prompt + outputs[prompt_length + 1:]
    print("prompt: ", prompt)
    print("outputs: ", outputs[prompt_length + 1:])
    print("outputs length:", len(outputs[prompt_length + 1:]))

    print("generated: ", generated)

    print("finished...")


if __name__ == '__main__':
    main()
