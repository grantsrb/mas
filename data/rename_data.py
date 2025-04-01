import sys
sys.path.append("../")
from utils import load_json, save_json

if __name__=="__main__":
    json_name = "multiobj_systematic_10000.json"
    demo = None
    resp = None
    trig = "trig_wordasst_word"
    done = None

    if len(sys.argv)>1:
        for arg in sys.argv[1:]:
            if ".json" in arg:
                json_name = arg
            elif "demo=" in arg:
                demo = arg.split("=")[-1]
            elif "resp=" in arg:
                resp = arg.split("=")[-1]
            elif "trig=" in arg:
                trig = arg.split("=")[-1]
            elif "done=" in arg:
                trig = arg.split("=")[-1]

    d = load_json(json_name)
    demo_toks = None
    if demo:
        demo_toks = {dd for dd in d["text"][-1].split(" ") if "demo_word" in dd}
    for i,seq in enumerate(d["text"]):
        if demo_toks:
            for tok in demo_toks:
                d["text"][i] = d["text"][i].replace(tok, demo)
        if resp:
            d["text"][i] = d["text"][i].replace("resp_word", resp)
        if trig:
            d["text"][i] = d["text"][i].replace("trig_word", trig)
        if done:
            d["text"][i] = d["text"][i].replace("done_word", done)
    save_json(d, json_name.split(".")[0]+ f"_wassistant.json")
