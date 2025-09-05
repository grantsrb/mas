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
        self.swap_varbs = None

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

    def trigger_fxn(self, x):
        """
        This is a function to call when the trigger token appears when
        updating the variables.
        """
        return x

    def post_intervention_fxn(self, varbs):
        return varbs

    def perform_intervention(self, varbs):
        if self.swap_varbs:
            for key in self.swap_varbs:
                if key not in self.ignore_keys:
                    varbs[key] = self.swap_varbs[key]
            varbs = self.post_intervention_fxn(varbs)
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
        for k in kwargs:
            if k in varbs: varbs[k] = kwargs[k]
        varbs = self.update_varbs(token_id, varbs, info=info)
        varbs = self.perform_intervention(varbs)
        self.clear_intervention()
        outp_token, tmask = self.get_token(varbs, info=info)
        return outp_token, varbs, tmask

class CountUpDown(CausalModel):
    def __init__(self,
            obj_count=None,
            min_count=1,
            max_count=20,
            hold_outs={4,9,14,17},
            unk_p=0.0,
            *args, **kwargs,
        ):
        super().__init__()
        self.min_count = min_count
        self.max_count = max_count
        self.hold_outs = hold_outs
        self.init_varbs_ = {
            "obj_count": obj_count,
            "unk_p": unk_p,
            "phase": -1,
            "count": 0,
        }
        self.ignore_keys = { "obj_count" }
        self.swap_varbs = None

    @property
    def init_varbs(self):
        varbs = copy.deepcopy(self.init_varbs_)
        if varbs["obj_count"] is None:
            varbs["obj_count"] = int(np.random.randint(
                self.min_count, self.max_count+1
            ))
            while varbs["obj_count"] in self.hold_outs:
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
            if token_id in info["trig_token_ids"]:
                varbs["phase"] = 1
                varbs["count"] = self.trigger_fxn(varbs["count"])
            elif token_id in info["demo_token_ids"]:
                varbs["count"] += 1
        else:
            if token_id in info["resp_token_ids"]:
                varbs["count"] -= 1
        return varbs

    def get_token(self, varbs, info, *args, **kwargs):
        if varbs["phase"]==-1:
            tmask = 0
            if np.random.random()<varbs.get("unk_p", 0):
                return info.get("unk_token_id", 8), tmask
            if varbs["count"]>=varbs["obj_count"]:
                tidx = int(np.random.randint(len(info["trig_token_ids"])))
                return info["trig_token_ids"][tidx], tmask
            didx = int(np.random.randint(len(info["demo_token_ids"])))
            return info["demo_token_ids"][didx], tmask
        else:
            if varbs["count"]<0:
                return info.get("pad_token_id", "<PAD>"), 0
            tmask = 1
            if varbs["count"]==0:
                return info["eos_token_id"], tmask
            ridx = int(np.random.randint(len(info["resp_token_ids"])))
            return info["resp_token_ids"][ridx], tmask

class CountUpUp(CountUpDown):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.init_varbs_["count"]
        self.init_varbs_["demo_count"] = 0
        self.init_varbs_["resp_count"] = 0
        self.ignore_keys = { "obj_count" }
        self.swap_varbs = None

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
            if np.random.random()<varbs.get("unk_p", 0):
                return info.get("unk_token_id", 8), tmask
            if varbs["demo_count"]>=varbs["obj_count"]:
                tidx = int(np.random.randint(len(info["trig_token_ids"])))
                return info["trig_token_ids"][tidx], tmask
            didx = int(np.random.randint(len(info["demo_token_ids"])))
            return info["demo_token_ids"][didx], tmask
        else:
            if varbs["resp_count"]>varbs["demo_count"]:
                return info.get("eos_token_id", "<PAD>"), 0
            tmask = 1
            if varbs["resp_count"]==varbs["demo_count"]:
                return info["eos_token_id"], tmask
            ridx = int(np.random.randint(len(info["resp_token_ids"])))
            return info["resp_token_ids"][ridx], tmask

class CountUpIncr(CountUpDown):
    """
    The CountUpIncr always uses the same increment in the demonstration phase.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_varbs_["max_count"] = kwargs.get("max_count", 20)
        self.init_varbs_["incr"] = 1/self.init_varbs_["max_count"]
        self.ignore_keys = { "obj_count" }
        self.swap_varbs = None

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
            if token_id in info["trig_token_ids"]:
                varbs["phase"] = 1
                varbs["incr"] = 1/max(varbs["count"],1)
                varbs["count"] = 0
            elif token_id in info["demo_token_ids"]:
                varbs["count"] += 1
        else:
            if token_id in info["resp_token_ids"]:
                varbs["count"] += varbs["incr"]*varbs["max_count"]
        return varbs

    def get_token(self, varbs, info, *args, **kwargs):
        if varbs["phase"]==-1:
            tmask = 0
            if np.random.random()<varbs.get("unk_p", 0):
                return info.get("unk_token_id", 8), tmask
            if varbs["count"]>=varbs["obj_count"]:
                tidx = int(np.random.randint(len(info["trig_token_ids"])))
                return info["trig_token_ids"][tidx], tmask
            didx = int(np.random.randint(len(info["demo_token_ids"])))
            return info["demo_token_ids"][didx], tmask
        else:
            if varbs["count"] < 0:
                return info.get("pad_token_id", "<PAD>"), 0
            tmask = 1
            # if the count is greater than or equal to the max_count,
            # then we return the eos token. We include a small offset
            # to avoid numerical issues with floating point precision.
            if varbs["count"]>=varbs["max_count"]-1/(2*varbs["max_count"]):
                return info["eos_token_id"], tmask
            ridx = int(np.random.randint(len(info["resp_token_ids"])))
            return info["resp_token_ids"][ridx], tmask


class IncrementUpUp(CountUpIncr):
    """
    This differs from the CountUpIncr in that it uses the existing
    incr value when performing interventions from the response phase
    to the demonstration phase. The CountUpIncr always uses the same
    increment in the demonstration phase.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        if varbs["phase"]==-1: # Demo Phase
            if token_id in info["trig_token_ids"]:
                varbs["phase"] = 1
                varbs["incr"] = 1/max(varbs["count"],1)
                varbs["count"] = 0
            elif token_id in info["demo_token_ids"]:
                varbs["count"] += varbs["incr"]*varbs["max_count"]
        else:
            if token_id in info["resp_token_ids"]:
                varbs["count"] += varbs["incr"]*varbs["max_count"]
        return varbs


class CountUpDownMod(CountUpDown):
    """
    This model counts some initial demo tokens and then takes the
    mod of that count to produce the response. 
    """
    def __init__(self, mod=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod = mod

    def trigger_fxn(self, x):
        return x%self.mod

    def post_intervention_fxn(self, varbs):
        if varbs["phase"]==1:
            varbs["count"] = self.trigger_fxn(varbs["count"])
        return varbs

class CountUpDownSquare(CountUpDown):
    """
    This model counts some initial demo tokens and then takes the
    square of that count to produce the response. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trigger_fxn(self, x):
        return x**2

class CountUpDownRound(CountUpDown):
    """
    This model counts some initial demo tokens and then rounds
    to the nearest Ns place.
    """
    def __init__(self, roundn=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roundn = roundn

    def trigger_fxn(self, x):
        return round(x/self.roundn)*self.roundn


class ArithmeticCmodel(CausalModel):
    def __init__(self,
            min_val=0,
            max_val=20,
            max_ops=20,
            blank_state=False,
            *args, **kwargs,
        ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.max_ops = max_ops
        # each of these are the real values, not token ids,
        # except for next_token_id
        self.init_varbs_ = {
            "remops": None, 
            "cumu_val": None,
            "op": None,
            "operand": None,
            "next_token_id": None,
        }
        self.swap_varbs = None

    @property
    def init_varbs(self):
        return copy.deepcopy(self.init_varbs_)

    def evaluate_cumu(self, varbs, info):
        cumu = varbs["cumu_val"]
        op = varbs["op"]
        operand = varbs["operand"]
        return eval( f"{cumu}{op}{operand}" )

    def sample_op(self, varbs):
        """Samples the evaluatable value"""
        cumu = varbs["cumu_val"]
        if cumu==self.max_val: op = "-"
        elif cumu==self.min_val: op = "+"
        else: op = "-" if np.random.random()>0.5 else "+"
        return op

    def sample_op_id(self, varbs, info):
        op = self.sample_op(varbs)
        idx = info["op_tokens"].index(op)
        return info["op_token_ids"][idx]

    def sample_operand(self, varbs):
        """Samples the evaluatable value"""
        cumu = varbs["cumu_val"]
        if varbs["op"]=="-":
            operand = np.random.randint(0,cumu-self.min_val+1)
        else:
            operand = np.random.randint(0,self.max_val-cumu+1)
        return str(operand)

    def sample_operand_id(self, varbs, info):
        operand = self.sample_operand(varbs)
        return self.get_num_tok_id(operand, info)

    def get_num_tok_id(self, num, info):
        """
        Converts the raw str number token to the token id
        """
        idx = info["number_tokens"].index(str(num))
        return info["number_token_ids"][idx]
    
    def sample_remops(self,):
        return int(np.random.randint(1,self.max_ops+1))

    def update_varbs(self,
            token_id,
            varbs,
            info,
            *args, **kwargs):
        try:
            if varbs is None: varbs = self.init_varbs
            if token_id in {info["eos_token_id"], info["pad_token_id"]}:
                varbs["remops"] = -1
                varbs["next_token_id"] = info["pad_token_id"]
                varbs["tmask"] = 0
                return varbs
            elif varbs["remops"] is None:
                varbs["remops"] = self.sample_remops()
                varbs["next_token_id"] = self.get_num_tok_id(
                    varbs["remops"],info)
                varbs["tmask"] = 0
            elif varbs["cumu_val"] is None:
                varbs["cumu_val"] = int(np.random.randint(
                    self.min_val, self.max_val+1))
                varbs["next_token_id"] = self.get_num_tok_id(
                    varbs["cumu_val"],info)
                varbs["tmask"] = 0
            elif token_id==info["equals_token_id"]:
                varbs["cumu_val"] = self.evaluate_cumu(varbs,info)
                varbs["next_token_id"] = self.get_num_tok_id(
                    varbs["cumu_val"],info)
                varbs["tmask"] = 1
            elif token_id==info["comma_token_id"]:
                varbs["remops"] -= 1
                varbs["next_token_id"] = self.sample_op_id(varbs, info)
                varbs["tmask"] = 0
            elif token_id in info["op_token_ids"]:
                idx = info["op_token_ids"].index(token_id)
                varbs["op"] = info["op_tokens"][idx]
                varbs["next_token_id"] = self.sample_operand_id(varbs, info)
                varbs["tmask"] = 0
            elif token_id in info["number_token_ids"]:
                varbs["tmask"] = 1
                idx = info["number_token_ids"].index(token_id)
                inpt = info["number_tokens"][idx]
                if varbs["op"] is None:
                    varbs["next_token_id"] = self.sample_op_id(varbs, info)
                    varbs["tmask"] = 0
                elif varbs["operand"] is None: # case that it is the operand
                    varbs["operand"] = inpt
                    varbs["next_token_id"] = info["equals_token_id"]
                else: # case that it is the cumu_val
                    varbs["op"] = None
                    varbs["operand"] = None
                    varbs["next_token_id"] = info["comma_token_id"]
        except:
            for k,v in varbs.items():
                print(k,v)
            assert False
        return varbs

    def get_token(self, varbs, info, *args, **kwargs):
        if varbs["remops"]<0:
            return info["pad_token_id"], 0
        elif varbs["next_token_id"]==info["comma_token_id"]:
            if varbs["remops"]==1:
                return info["eos_token_id"],varbs["tmask"]
        return varbs["next_token_id"],varbs["tmask"]


