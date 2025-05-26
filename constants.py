# Store prompts that worked here
PROMPTS = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": (
        "user_word For each trial, I am going to say \"demo_word0\" some number of times "
        "and then I want you to respond bysaying \"resp_word\" the exact same number of "
        "times that I said \"demo_word0\" in that trial. Please say \"done_word\" at the "
        "end of your response to indicate that you are finished. Do not say or think anything else.\n"
        "Example: user_word demo_word0 trig_wordasst_word resp_word done_word \n"
        "Example: user_word demo_word0 demo_word0 trig_wordasst_word resp_word resp_word done_word \n"
        "Begin Trial: user_word "
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (
        "user_word For each trial, I am going to say demo_word0 some number "
        "of times and then I want you to respond by saying resp_word the exact "
        "same number of times that I said demo_word0 in that trial. Please say "
        "done_word at the end of your response to indicate that you are finished.\n"
        "Example: user_word demo_word0 trig_wordasst_word resp_word done_word \n"
        "Example: user_word demo_word0 demo_word0 trig_wordasst_word resp_word "
        "resp_word done_word \nBegin Trial: user_word "
     ),

    "gpt2": (
        "user_word For each trial, I am going to say \"demo_word0\" some number "
        "of times and then I want you to respond bysaying \"resp_word\" the exact "
        "same number of times that I said \"demo_word0\" in that trial. Please say "
        "\"done_word\" at the end of your response to indicate that you are finished. "
        "Do not say or think anything else.\nExample: user_word demo_word0 trig_wordasst_word "
        "resp_word done_word \nExample: user_word demo_word0 demo_word0 trig_wordasst_word "
        "resp_word resp_word done_word \nBegin Trial: user_word "
    ),
}

DEFAULT_REPLACEMENTS = {
    "demo_word0": "D0",
    "demo_word1": "D1",
    "demo_word2": "D2",
    "asst_word": "",
    "user_word": "",
    "trig_word": "T",
    "resp_word": "R",
    "done_word": "E",
}

REPLACEMENTS = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "demo_word0": "nut",
        "demo_word1": "nut",
        "demo_word2": "nut",
        "trig_word": ":",
        "asst_word": "",
        "user_word": "User:",
        "resp_word": "seed",
        "done_word": "stop",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "demo_word0": "nut",
        "demo_word1": "nut",
        "demo_word2": "nut",
        "trig_word": ":",
        "asst_word": "",
        "user_word": "User:",
        "resp_word": "seed",
        "done_word": "stop",
    },
    "gpt2": {
        "demo_word0": "nut",
        "demo_word1": "nut",
        "demo_word2": "nut",
        "trig_word": ":",
        "asst_word": "",
        "user_word": "User:",
        "resp_word": "seed",
        "done_word": "stop",
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
