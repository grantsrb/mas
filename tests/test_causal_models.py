import unittest

import sys
wd = sys.path[0]
sys.path = [wd.split("tests")[0]] + sys.path 
from causal_models import CountUpIncr, CountUpDown, IncrementUpUp
from tasks import MultiObject, SingleObject, SameObject

def is_correct_sameobj_sequence(seq, expected_count):
    demo_count = 0
    resp_count = 0
    triggered = False
    for token in seq:
        if token in {"D"}:
            demo_count += not triggered
            resp_count += triggered
        if token == "T":
            triggered = True
    if seq[-1] != "E":
        return False
    return resp_count == expected_count and demo_count == expected_count

def is_correct_sequence(seq, expected_count):
    demo_count = 0
    resp_count = 0
    triggered = False
    for token in seq:
        if token in {"D0", "D1", "D2", "D"}:
            demo_count += 1
            if triggered:
                return False
        if token == "T":
            triggered = True
        if token == "R":
            if not triggered:
                return False
            resp_count += 1
    if seq[-1] != "E":
        return False
    return resp_count == expected_count and demo_count == expected_count

class TestCountUpIncr(unittest.TestCase):
    def setUp(self):
        self.model = CountUpIncr(max_count=20)

    def test_initialization(self):
        self.assertEqual(self.model.init_varbs["max_count"], 20)
        self.assertEqual(self.model.init_varbs["incr"], 0.05)

    def test_valid_sameobj_sequence(self):
        task_obj = SameObject()
        task_obj.cmodel = self.model
        seq, tmask, varbs = task_obj.generate_sample(
            obj_count=3,
        )
        for i in range(1,20):
            seq, tmask, varbs = task_obj.generate_sample(
                obj_count=i,
            )
            correct = is_correct_sameobj_sequence(seq, i)
            self.assertTrue(correct, f"Failed for obj_count={i}. Sequence: {seq}")

    def test_valid_multiobj_sequence(self):
        task_obj = SingleObject()
        task_obj.cmodel = self.model
        seq, tmask, varbs = task_obj.generate_sample(
            obj_count=3,
        )
        for i in range(1,20):
            seq, tmask, varbs = task_obj.generate_sample(
                obj_count=i,
            )
            correct = is_correct_sequence(seq, i)
            self.assertTrue(correct, f"Failed for obj_count={i}. Sequence: {seq}")

    def test_valid_multiobj_sequence(self):
        task_obj = MultiObject()
        task_obj.cmodel = self.model
        seq, tmask, varbs = task_obj.generate_sample(
            obj_count=3,
        )
        for i in range(1,20):
            seq, tmask, varbs = task_obj.generate_sample(
                obj_count=i,
            )
            correct = is_correct_sequence(seq, i)
            self.assertTrue(correct, f"Failed for obj_count={i}. Sequence: {seq}")

class TestIncrementUpUp(unittest.TestCase):
    def setUp(self):
        self.model = IncrementUpUp(max_count=20)

    def test_initialization(self):
        self.assertEqual(self.model.init_varbs["max_count"], 20)
        self.assertEqual(self.model.init_varbs["incr"], 0.05)

    def test_valid_sameobj_sequence(self):
        task_obj = SameObject()
        task_obj.cmodel = self.model
        seq, tmask, varbs = task_obj.generate_sample(
            obj_count=3,
        )
        for i in range(1,20):
            seq, tmask, varbs = task_obj.generate_sample(
                obj_count=i,
            )
            correct = is_correct_sameobj_sequence(seq, i)
            self.assertTrue(correct, f"Failed for obj_count={i}. Sequence: {seq}")

    def test_valid_multiobj_sequence(self):
        task_obj = SingleObject()
        task_obj.cmodel = self.model
        seq, tmask, varbs = task_obj.generate_sample(
            obj_count=3,
        )
        for i in range(1,20):
            seq, tmask, varbs = task_obj.generate_sample(
                obj_count=i,
            )
            correct = is_correct_sequence(seq, i)
            self.assertTrue(correct, f"Failed for obj_count={i}. Sequence: {seq}")

    def test_valid_multiobj_sequence(self):
        task_obj = MultiObject()
        task_obj.cmodel = self.model
        seq, tmask, varbs = task_obj.generate_sample(
            obj_count=3,
        )
        for i in range(1,20):
            seq, tmask, varbs = task_obj.generate_sample(
                obj_count=i,
            )
            correct = is_correct_sequence(seq, i)
            self.assertTrue(correct, f"Failed for obj_count={i}. Sequence: {seq}")

if __name__ == "__main__":
    unittest.main()
