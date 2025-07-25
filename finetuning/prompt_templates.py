PROMPT_TEMPLATES = {
    "anitamaxvim/jigsaw-toxic-comments": "Assistant: {prompt}",
    "Anthropic/hh-rlhf": "{prompt}",
    "lmsys/toxic-chat": "Human: {prompt}\nAssistant:"
}

PROMPTS = {
    "toxic": {
        "anitamaxvim/jigsaw-toxic-comments": "",
        "Anthropic/hh-rlhf": "",
        "lmsys/toxic-chat": "",
    },
    "nontoxic": {
        "anitamaxvim/jigsaw-toxic-comments": "",
        "Anthropic/hh-rlhf": "",
        "lmsys/toxic-chat": "",
    },
    "both": {
        "anitamaxvim/jigsaw-toxic-comments": "",
        "Anthropic/hh-rlhf": "",
        "lmsys/toxic-chat": "",
    },
}