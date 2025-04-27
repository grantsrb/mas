import unittest

try:
    from causal_models import CountUpDown, CountUpDownMod, CountUpDownSquare, CountUpDownRound
    from tasks import (
        MultiObject, SingleObject, SameObject,
        MultiObjectMod, SingleObjectMod, SameObjectMod,
        MultiObjectSquare, SingleObjectSquare, SameObjectSquare,
        MultiObjectRound, SingleObjectRound, SameObjectRound
    )
except ImportError:
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from causal_models import CountUpDown, CountUpDownMod, CountUpDownSquare, CountUpDownRound
    from tasks import (
        MultiObject, SingleObject, SameObject,
        MultiObjectMod, SingleObjectMod, SameObjectMod,
        MultiObjectSquare, SingleObjectSquare, SameObjectSquare,
        MultiObjectRound, SingleObjectRound, SameObjectRound
    )


class TestTasks(unittest.TestCase):
    def test_multi_object(self):
        task = MultiObject()
        self.assertIsInstance(task.cmodel, CountUpDown)
        self.assertEqual(task.info["bos_token"], "B")
        self.assertIn("D0", task.info["demo_tokens"])

    def test_single_object(self):
        task = SingleObject()
        self.assertIsInstance(task.cmodel, CountUpDown)
        self.assertEqual(task.info["bos_token"], "B")
        self.assertEqual(task.info["demo_tokens"], ["D"])

    def test_same_object(self):
        task = SameObject()
        self.assertIsInstance(task.cmodel, CountUpDown)
        self.assertEqual(task.info["bos_token"], "B")
        self.assertEqual(task.info["resp_tokens"], ["D"])

    def test_multi_object_mod(self):
        task = MultiObjectMod()
        self.assertIsInstance(task.cmodel, CountUpDownMod)

    def test_single_object_mod(self):
        task = SingleObjectMod()
        self.assertIsInstance(task.cmodel, CountUpDownMod)

    def test_same_object_mod(self):
        task = SameObjectMod()
        self.assertIsInstance(task.cmodel, CountUpDownMod)

    def test_multi_object_square(self):
        task = MultiObjectSquare()
        self.assertIsInstance(task.cmodel, CountUpDownSquare)

    def test_single_object_square(self):
        task = SingleObjectSquare()
        self.assertIsInstance(task.cmodel, CountUpDownSquare)

    def test_same_object_square(self):
        task = SameObjectSquare()
        self.assertIsInstance(task.cmodel, CountUpDownSquare)

    def test_multi_object_round(self):
        task = MultiObjectRound()
        self.assertIsInstance(task.cmodel, CountUpDownRound)

    def test_single_object_round(self):
        task = SingleObjectRound()
        self.assertIsInstance(task.cmodel, CountUpDownRound)

    def test_same_object_round(self):
        task = SameObjectRound()
        self.assertIsInstance(task.cmodel, CountUpDownRound)

    def test_multi_object_generate_sample(self):
        task = MultiObject()
        seq, tmask, varbs = task.generate_sample(obj_count=3)
        print("MultiObject sample:", seq)
        print("MultiObject Tmask :", tmask)
        print("MultiObject Meta  :", varbs[-1])

    def test_multi_object_generate_samples(self):
        task = MultiObject()
        n_samples = 3
        seqs, tmasks, metas = task.generate_samples(n_samples=n_samples)
        print("MultiObject samples:", seqs)
        self.assertEqual(len(seqs), n_samples)
        self.assertEqual(len(tmasks), n_samples)
        self.assertEqual(len(metas), n_samples)


if __name__ == "__main__":
    unittest.main()