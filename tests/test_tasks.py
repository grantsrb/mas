import unittest

try:
    from causal_models import CountUpDown, CountUpDownMod, CountUpDownSquare, CountUpDownRound, ArithmeticCmodel
    from tasks import (
        MultiObject, SingleObject, SameObject,
        MultiObjectMod, SingleObjectMod, SameObjectMod,
        MultiObjectSquare, SingleObjectSquare, SameObjectSquare,
        MultiObjectRound, SingleObjectRound, SameObjectRound,
        Arithmetic,
    )
except ImportError:
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from causal_models import CountUpDown, CountUpDownMod, CountUpDownSquare, CountUpDownRound, ArithmeticCmodel
    from tasks import (
        MultiObject, SingleObject, SameObject,
        MultiObjectMod, SingleObjectMod, SameObjectMod,
        MultiObjectSquare, SingleObjectSquare, SameObjectSquare,
        MultiObjectRound, SingleObjectRound, SameObjectRound,
        Arithmetic,
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

    def test_arithmetic(self):
        task = Arithmetic()
        self.assertIsInstance(task.cmodel, ArithmeticCmodel)

    def test_arithmetic_equals_count(self):
        task = Arithmetic()
        seq, tmask, varbs = task.generate_sample()
        n_ops = int(seq[0])
        n_eqs = sum([1 if s=="=" else 0 for s in seq])
        self.assertEqual(n_ops, n_eqs)

    def test_arithmetic_correct_seq(self):
        task = Arithmetic()
        seq, tmask, varbs = task.generate_sample()
        segs = ("".join(seq)).split(",")
        cumu_val = eval("".join(seq[1:4]))
        self.assertEqual(int(segs[0].split("=")[-1]), cumu_val)
        for seg in segs[1:]:
            ops = seg.split("=")[0]
            new_cumu = int(seg.split("=")[-1].replace("E",""))
            cumu_val = eval(str(cumu_val)+ops)
            self.assertEqual(cumu_val, new_cumu)

    def test_arithmetic_generate_sample(self):
        task = Arithmetic()
        seq, tmask, varbs = task.generate_sample()
        print("Arithmetic sample:", seq)
        print("Arithmetic Tmask :", tmask)
        print("Arithmetic Meta  :", varbs[-1])

    def test_arithmetic_generate_samples(self):
        task = Arithmetic()
        n_samples = 3
        seqs, tmasks, metas = task.generate_samples(n_samples=n_samples)
        print("Arithmetic samples:", seqs)
        self.assertEqual(len(seqs), n_samples)
        self.assertEqual(len(tmasks), n_samples)
        self.assertEqual(len(metas), n_samples)


if __name__ == "__main__":
    unittest.main()
