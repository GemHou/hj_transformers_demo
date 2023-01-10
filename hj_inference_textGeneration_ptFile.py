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

OUTPUT_LENGTH = 8


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

    full_acid = "VILPNNDRHQITDTTNGHYAPVTYIQVEAPTGTFIASGVVVGKDTLLTNKHVVDATHGDPHALKAFPSAINQDNYPNGGFTAEQITKYSGEGDLAIVKFSPNEQNKHIGEVVKPATMSNNAETQTNQNITVTGYPGDKPVATMWESKGKITYLKGEAMQYDLSTTGGNSGSPVFNEKNEVIGIHWGGVPNEFNGAVFINENVRNFLKQNIEDINFANDDQPNNPDNPDNPNNPDNPNNPDNPNNPDEPNNPDNPNNPDNPDNGDNNNSDNPDAA"

    init_obs_acid = full_acid[:int(len(full_acid)*0.3)]  # int(len(full_acid)*0.3)

    true_acid = full_acid[int(len(full_acid)*0.3):]

    # print("true_acid: ", true_acid)

    return PADDING_TEXT, init_obs_acid, full_acid, true_acid


def test_success_rate(PADDING_TEXT, init_obs_acid, model, tokenizer, true_acid):
    obs_acid = init_obs_acid
    success_num = 0
    fail_num = 0
    for true_letter in true_acid:
        # print("true_letter: ", true_letter)
        inputs = tokenizer(PADDING_TEXT + obs_acid, add_special_tokens=False, return_tensors="pt")["input_ids"]
        inputs_length = inputs[0].shape[0]
        prompt_length = len(tokenizer.decode(inputs[0]))
        outputs = model.generate(inputs, attention_mask=torch.tensor([[1] * inputs.shape[1]]), max_length=inputs_length + 1, do_sample=True, top_p=0.95, top_k=60)
        outputs = outputs[0]
        outputs = tokenizer.decode(outputs)
        outputs_length = len(outputs)
        # print("outputs_length: ", outputs_length)
        outputs_letter = outputs[prompt_length:]
        if outputs_letter[0] == " ":
            outputs_letter = outputs_letter[1]
        else:
            outputs_letter = outputs_letter[0]
        assert outputs_letter != " "
        # print("outputs_letter: ", outputs_letter)
        if outputs_letter == true_letter:
            # print("success!")
            success_num += 1
        else:
            # print("fail!")
            fail_num += 1
        obs_acid = obs_acid + true_letter
    success_rate = success_num / (success_num + fail_num)
    return success_rate


def main():
    model_name = "distilgpt2"  # bert-base-cased-finetuned-mrpc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)  # bert-base-cased
    model.load_state_dict(torch.load("../para_temp.pt"))

    PADDING_TEXT, init_obs_acid, full_acid, true_acid = prepare_task()

    inputs = tokenizer(PADDING_TEXT + init_obs_acid, add_special_tokens=False, return_tensors="pt")[
        "input_ids"]  # PADDING_TEXT +

    inputs_length = inputs[0].shape[0]
    print("inputs_length: ", inputs_length)
    prompt_length = len(tokenizer.decode(inputs[0]))
    print("prompt_length: ", prompt_length)
    inputs_dict = dict()
    inputs_dict["inputs"] = inputs
    inputs_dict["attention_mask"] = torch.tensor([[1] * inputs.shape[1]])

    print("forward... waiting...")
    start_time = time.time()
    outputs = model.generate(inputs, attention_mask=torch.tensor([[1] * inputs.shape[1]]), max_length=inputs_length+OUTPUT_LENGTH, do_sample=True, top_p=0.95, top_k=60)  # max_length=250
    print("forward time: ", time.time()-start_time)

    outputs = outputs[0]
    outputs = tokenizer.decode(outputs)
    generated = init_obs_acid + outputs[prompt_length + 1:]

    print("prompt: ", init_obs_acid)
    print("outputs: ", outputs[prompt_length + 1:])
    print("outputs length:", int((len(outputs[prompt_length + 1:]) + 1) / 2))
    print("generated: ", generated)

    success_rate = test_success_rate(PADDING_TEXT, init_obs_acid, model, tokenizer, true_acid)
    print("success_rate: ", success_rate)

    print("finished...")


if __name__ == '__main__':
    main()
