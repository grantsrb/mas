import copy

import causal_models
from utils import run_cmodel_to_completion

multiobj_info = {
    "bos_token": "B",
    "eos_token": "E",
    "pad_token": "P",
    "demo_tokens": ["D0", "D1", "D2"],
    "resp_tokens": ["R"],
    "trig_tokens": ["T"],
}

singleobj_info = copy.deepcopy(multiobj_info)
singleobj_info["demo_tokens"] = ["D"]

sameobj_info = copy.deepcopy(singleobj_info)
sameobj_info["resp_tokens"] = ["D"]

class Task:
    def __init__(self, cmodel, info=None, *args, **kwargs):
        self.cmodel = cmodel
        self.info = info if info is not None else dict()
        self.bos_token = self.info.get("bos_token", "B")

    def prep_info(self, info):
        # Need to use _id to integrate with the CausalModels
        for k in list(info.keys()):
            if "tokens" in k:
                kid = k.replace("tokens", "token_ids")
                info[kid] = copy.deepcopy(info[k])
            elif "token" in k:
                kid = k + "_id"
                info[kid] = copy.deepcopy(info[k])
        return info

    def get_info(self):
        return {
            k:copy.deepcopy(v) for k,v in self.info.items() if "_id" not in k
        }

    def generate_sample(self, *args, **kwargs):
        seq, tmask, varbs = run_cmodel_to_completion(
            cmodel=self.cmodel,
            inpt_token=self.bos_token,
            info=self.info,
            end_tokens={self.info.get("eos_token", "E"), None},
            *args, **kwargs,
        )
        return seq, tmask, varbs

    def generate_samples(self, n_samples, *args, **kwargs):
        seqs, tmasks, metas = [], [], []
        for si in range(n_samples):
            seq, tmask, varbs = self.generate_sample(*args, **kwargs)
            seqs.append(seq)
            tmasks.append(tmask)
            metas.append(varbs[-1])
        return seqs, tmasks, metas

################################################################
# Numeric Equivlance Tasks
################################################################
class NumericEquivalence(Task):
    def __init__(self, info, *args, **kwargs):
        self.cmodel = causal_models.CountUpDown(*args, **kwargs)
        self.info = info
        self.bos_token = self.info["bos_token_id"]

class MultiObject(NumericEquivalence):
    def __init__(self, info=None, *args, **kwargs):
        if info is None:
            info = copy.deepcopy(multiobj_info)
            info = self.prep_info(info)
        super().__init__(info=info, *args, **kwargs)

class SingleObject(NumericEquivalence):
    def __init__(self, info=None, *args, **kwargs):
        if info is None:
            info = copy.deepcopy(singleobj_info)
            info = self.prep_info(info)
        super().__init__(info=info, *args, **kwargs)

class SameObject(NumericEquivalence):
    def __init__(self, info=None, *args, **kwargs):
        if info is None:
            info = copy.deepcopy(sameobj_info)
            info = self.prep_info(info)
        super().__init__(info=info, *args, **kwargs)

################################################################
# Mod Tasks
################################################################
class MultiObjectMod(MultiObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownMod(*args, **kwargs)

class SingleObjectMod(SingleObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownMod(*args, **kwargs)

class SameObjectMod(SameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownMod(*args, **kwargs)

################################################################
# Square Tasks
################################################################
class MultiObjectSquare(MultiObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownSquare(*args, **kwargs)

class SingleObjectSquare(SingleObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownSquare(*args, **kwargs)

class SameObjectSquare(SameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownSquare(*args, **kwargs)

################################################################
# Round Tasks
################################################################
class MultiObjectRound(MultiObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownRound(*args, **kwargs)

class SingleObjectRound(SingleObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownRound(*args, **kwargs)

class SameObjectRound(SameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmodel = causal_models.CountUpDownRound(*args, **kwargs)

