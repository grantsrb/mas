"""
This script finds the missing trainings from a given hyperparams.json
and a hyperranges.json. It searches the experpiment folder to find all
the existing trainings that are included in the parameter specifications.
It then determinew which hyperparameter combos still need to be trained.

If used in the following way, it will create a new hyperranges json
called missingranges.json that can be used with the hyperparams.json
to complete the training:

$ python3 collect_missing_trainings.py hyperparams.json hyperranges.json

Alternatively use the function `find_missing_trainings` in your own
script.
"""

from collections import deque, namedtuple
import sys
import os
try:
    import dl_utils.save_io as save_io
except:
    sys.path.append("../")
    import dl_utils.save_io as save_io
from dl_utils.training import fill_hyper_q

def find_missing_trainings(hyps, hranges):
    """
    Finds the missing trainings and constructs a new hranges that can
    be used with hyps to complete the trainings.

    Argumens:
        hyps: dict
            dict of hyperparameters
        hranges: dict
            a dict of lists corresponding to hyperparameter values to
            be searched over.
    Returns:
        hranges: dict
            updated set of hyperranges to complete the training
    """
    if "exp_folder" in hyps:
        exp_folder = hyps["exp_folder"]
    else:
        exp_folder = os.path.join(
            hyps.get("save_root", "./"), hyps["exp_name"]
        )
    model_folders = set(save_io.get_model_folders(
        exp_folder, incl_full_path=True, incl_empty=False
    ))

    hyper_q = deque()
    hyper_q = fill_hyper_q(hyps, hranges, list(hranges.keys()),
                                          hyper_q, idx=0)
    keys = []
    for k in hranges.keys():
        if type(hranges[k])==type(dict()):
            for kk in hranges[k].keys(): keys.append(kk)
        else: keys.append(k)
    Combo = namedtuple("Combo", " ".join(keys))
    combos = set()
    for h in hyper_q:
        combos.add( Combo(*[h[k] for k in keys]) )

    for folder in model_folders:
        h = save_io.get_hyps(folder)
        combo = Combo(*[h[k] for k in keys])
        if combo in combos: combos.remove(combo)

    hranges = {k: [] for k in keys}
    for combo in combos:
        for k in keys:
            hranges[k].append(getattr(combo,k))
    return {"pairs": hranges}

if __name__=="__main__":
    hyps = save_io.load_json(sys.argv[1])
    hranges = save_io.load_json(sys.argv[2])
    hranges = find_missing_trainings(hyps,hranges)
    print(hranges)
    fname = "missingranges.json"
    save_io.save_json(hranges, fname)
    print("Saved to", fname)
