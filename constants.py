# Store prompts that worked here
PROMPTS = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": (
        f"user_word For each trial, I am going to "
        f"say \"D0\" some number of times and then I want you to respond by "
        f"saying \"R\" the exact same number of times that I said \"D0\" in "
        f"that trial. Please say \"E\" at the end of your response to indicate "
        f"that you are finished. Do not say or think anything else. "
        f"\nexample: user_word D0 T R E "
        f"\nexample: user_word D0 D0 T R R E "
        f"\nexample: user_word D0 T R E "
        f"\nexample: user_word"
    ),

    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (
        f"user_word For each trial, I am going to "
        f"say \"D0\" some number of times and then I want you to respond by "
        f"saying \"R\" the exact same number of times that I said \"D0\" in "
        f"that trial. Please say \"E\" at the end of your response to indicate "
        f"that you are finished. Do not say or think anything else. "
        f"\nexample: user_word D0 T R E "
        f"\nexample: user_word D0 D0 T R R E "
        f"\nexample: user_word D0 T R E "
        f"\nexample: user_word"
     ),

    "gpt2": (
        "user_word For each trial, I am going to say \"D0\" some number "
        "of times and then I want you to respond by saying \"R\" the exact "
        "same number of times that I said \"D0\" in that trial. Please say "
        "\"E\" at the end of your response to indicate that you are finished. "
        "Do not say or think anything else.\nexample: user_word D0 T "
        "R E \nexample: user_word D0 D0 T "
        "R R E \nBegin trial: user_word "
    ),
}

DEFAULT_REPLACEMENTS = {
    "D0": "D0",
    "D1": "D1",
    "D2": "D2",
    "asst_word": "",
    "user_word": "",
    "T": "T",
    "R": "R",
    "E": "E",
}

REPLACEMENTS = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "D0": "item",
        "D1": "item",
        "D2": "item",
        "T": ":",
        "asst_word": "",
        "user_word": "",
        "R": "object",
        "E": "stop",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "D0": "item",
        "D1": "item",
        "D2": "item",
        "T": ":",
        "asst_word": "",
        "user_word": "",
        "R": "object",
        "E": "stop",
    },
    "gpt2": {
        "D0": "item",
        "D1": "item",
        "D2": "item",
        "T": ":",
        "asst_word": "",
        "user_word": "User:",
        "R": "object",
        "E": "stop",
    },
}

PADDING_SIDES = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "left",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "left",
    "gpt2": "left",
}

TASK2CMODEL = {
    "NumericEquivalence": "CountUpDown",
    "MultiObject": "CountUpDown",
    "SingleObject": "CountUpDown",
    "SameObject": "CountUpDown",

    "MultiObjectMod": "CountUpDownMod",
    "SingleObjectMod": "CountUpDownMod",
    "SameObjectMod": "CountUpDownMod",

    "MultiObjectSquare": "CountUpDownSquare",
    "SingleObjectSquare": "CountUpDownSquare",
    "SameObjectSquare": "CountUpDownSquare",

    "MultiObjectRound": "CountUpDownRound",
    "SingleObjectRound": "CountUpDownRound",
    "SameObjectRound": "CountUpDownRound",

    "Arithmetic": "Arithmetic",
}

if __name__=="__main__":
    p = PROMPTS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
    r = REPLACEMENTS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
    for k in r:
        p = p.replace(k, r[k])
    print(p)
