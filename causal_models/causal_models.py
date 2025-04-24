import copy
import numpy as np

"""
Important to remember that causal models should be designed
such that interventions occur alongside the input token ids.
This is contrasted against interventions that occur alongside
output token ids. Also, be sure to use token ids rather than
token strings.
"""

class CausalModel:
    def __init__(self,):
        self.init_varbs_ = {}
        self.ignore_keys = {} # optionally specify variable names to
            # ignore for interventions. Allows us to use "full" as a
            # variable name to swap multiple specific variables

    @property
    def init_varbs(self):
        return copy.deepcopy(self.init_varbs_)

    def queue_intervention(self, swap_varbs):
        """
        Args:
            swap_varbs: dict
                a dict of the values that need to be replaced in the
                intervention
        """
        self.swap_varbs = swap_varbs

    def perform_intervention(self, varbs):
        if self.swap_varbs:
            for key in self.swap_varbs:
                if key not in self.ignore_keys:
                    varbs[key] = self.swap_varbs
        return varbs

    def clear_intervention(self):
        self.swap_varbs = None

    def update_varbs(self, token_id, varbs, *args, **kwargs):
        if varbs is None: return self.init_varbs
        return copy.deepcopy(varbs)

    def get_token(self, varbs, *args, **kwargs):
        token = None
        tmask = None
        return token, tmask

    def __call__(self, token_id, varbs, info=None, *args, **kwargs):
        varbs = self.update_varbs(token_id, varbs, info=info)
        varbs = self.perform_intervention(varbs)
        self.clear_intervention()
        outp_token, tmask = self.get_token(varbs, info=info)
        return outp_token, varbs, tmask

def CountUpDown(CausalModel):
    def __init__(self, obj_count=None, max_count=20, min_count=1):
        super().__init__()
        self.max_count = max_count
        self.min_count = min_count
        self.init_varbs_ = {
            "obj_count": obj_count,
            "phase": -1,
            "count": 0,
        }
        self.ignore_keys = { "obj_count", }

    @property
    def init_varbs(self):
        varbs = copy.deepcopy(self.init_varbs_)
        if varbs["obj_count"] is None:
            varbs["obj_count"] = int(np.random.randint(
                self.min_count, self.max_count+1
            ))
        return varbs

    def update_varbs(self,
            token_id,
            varbs,
            info,
            *args, **kwargs):
        if varbs is None: varbs = self.init_varbs
        if token_id in {info["eos_token_id"], info["pad_token_id"]}:
            varbs["count"] = -1
            varbs["phase"] = 1
            return varbs
        if varbs["phase"]==-1:
            if token_id in info["trig_token_idss"]:
                varbs["phase"] = 1
            elif token_id in info["demo_token_ids"]:
                varbs["count"] += 1
        else:
            if token_id in info["resp_token_ids"]:
                varbs["count"] -= 1
        return varbs

    def get_token(self, varbs, info, *args, **kwargs):
        if varbs["phase"]==-1:
            tmask = 0
            if varbs["count"]>=varbs["obj_count"]:
                tidx = np.random.randint(len(info["trig_token_ids"]))
                return info["trig_token_ids"][tidx], tmask
            didx = np.random.randint(len(info["demo_token_ids"]))
            return info["demo_token_ids"][didx], tmask
        if varbs["count"]<0:
            return info.get("pad_token_id", "<PAD>"), 0
        tmask = 1
        if varbs["count"]==0:
            return info["eos_token_id"], tmask
        ridx = int(np.random.randint(len(info["resp_token_ids"])))
        return info["resp_token_ids"][ridx], tmask

def CountUpUp(CountUpDown):
    def __init__(self, obj_count=None, max_count=20, min_count=1):
        super().__init__()
        self.max_count = max_count
        self.min_count = min_count
        self.init_varbs_ = {
            "obj_count": obj_count,
            "phase": -1,
            "demo_count": 0,
            "resp_count": 0,
        }
        self.ignore_keys = { "obj_count", }

    def update_varbs(self,
            token_id,
            varbs,
            info,
            *args, **kwargs):
        if varbs is None: varbs = self.init_varbs
        if token_id in {info["eos_token_id"], info["pad_token_id"]}:
            varbs["resp_count"] = varbs["demo_count"]+1
            varbs["phase"] = 1
            return varbs
        if varbs["phase"]==-1:
            if token_id in info["trig_token_ids"]:
                varbs["phase"] = 1
            elif token_id in info["demo_token_ids"]:
                varbs["demo_count"] += 1
        else:
            if token_id in info["resp_token_ids"]:
                varbs["resp_count"] += 1
        return varbs

    def get_token(self, varbs, info, *args, **kwargs):
        if varbs["phase"]==-1:
            tmask = 0
            if varbs["demo_count"]>=varbs["obj_count"]:
                tidx = np.random.randint(len(info["trig_token_ids"]))
                return info["trig_token_ids"][tidx], tmask
            didx = np.random.randint(len(info["demo_token_ids"]))
            return info["demo_token_ids"][didx], tmask
        if varbs["resp_count"]>varbs["demo_count"]:
            return info.get("eos_token_id", "<PAD>"), 0
        tmask = 1
        if varbs["resp_count"]==varbs["demo_count"]:
            return info["eos_token_id"], tmask
        ridx = int(np.random.randint(len(info["resp_token_ids"])))
        return info["resp_token_ids"][ridx], tmask