from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class DecayScheduler(_LRScheduler):
    """
    Code adapted from https://kikaben.com/transformers-training-details/
    to have a more flexible decay rate and relationship between
    warmup steps and the maximum learning rate.
    """
    @staticmethod
    def calc_lr(step,
            warmup_steps,
            max_lr=0.005,
            min_lr=1e-7,
            decay_exp=0.5,
        ):
        """
        Args:
            warmup_steps: int
            min_lr: float
                sets a lower bound on the learning rate. the lr will
                never drop below this value
            max_lr: float
                the maximum learning rate. This learning rate will
                be returned at the end of the warmup.
            decay_exp: float
                an exponent dictating the rate of decay of the learning
                rate following the warmup.
        """
        scale = max_lr * warmup_steps**(decay_exp)
        warmup = scale * step / warmup_steps**(1+decay_exp)
        reg = np.maximum(scale*step**(-decay_exp), min_lr)
        return np.minimum(reg, warmup)

    def __init__(self, 
                 optimizer,
                 warmup_steps: int=100,
                 last_epoch: int=-1,
                 verbose: bool=False,
                 min_lr: float=1e-10,
                 lr: float=1,
                 lr_decay_exp=0.25,
                 *args, **kwargs) -> None:
        """
        Args:
            warmup_steps: int
            min_lr: float
                sets a lower bound on the learning rate. the lr will
                never drop below this value
            lr: float
                the maximum learning rate. This learning rate will
                be returned at the end of the warmup.
            lr_decay_exp: float
                an exponent dictating the rate of decay of the learning
                rate following the warmup.
        """
        self.max_lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.lr_decay_exp = lr_decay_exp
        self.num_param_groups = len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = DecayScheduler.calc_lr(
            self._step_count,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            decay_exp=self.lr_decay_exp,
        )
        return [float(lr)] * self.num_param_groups

class PlateauTracker:
    def __init__(self, **kwargs):
        self.patience = kwargs.get("patience", 20)
        self.plateau =  kwargs.get("plateau", 0.01)
        self.measure =  kwargs.get("measure", "acc")
        self.reset()

    def reset(self):
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_measure = self.best_val_acc if self.measure=="acc" else\
                            self.best_val_loss
        self.counter = 0

    def update(self, val_loss, val_acc):
        if self.patience is None or not self.plateau or self.plateau<=0:
            return False
        measure = val_acc if self.measure=="acc" else val_loss
        if measure < (self.best_measure-self.plateau):
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_measure = val_acc if self.measure=="acc" else val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


