import unittest
import torch
try:
    from fca import ( FunctionalComponentAnalysis )
except ImportError:
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from fca import ( FunctionalComponentAnalysis )

def all_orthogonal(vecs, tolerance=0.0001):
    for i, vec1 in enumerate(vecs):
        for j, vec2 in enumerate(vecs):
            if i != j:
                if torch.dot(vec1, vec2).item() > tolerance:
                    print(i,j)
                    print("Dot:", torch.dot(vec1, vec2).item())
                    return False
    return True

class TestFunctionalComponentAnalysis(unittest.TestCase):

    def setUp(self):
        self.size = 10
        self.fca = FunctionalComponentAnalysis(size=self.size)

    def test_initialization(self):
        self.assertEqual(self.fca.size, self.size)
        self.assertEqual(len(self.fca.parameters_list), 1)
        self.assertFalse(self.fca.is_fixed)

    def test_add_new_axis_parameter(self):
        param = self.fca.add_new_axis_parameter()
        self.assertEqual(len(self.fca.parameters_list), 2)
        self.assertTrue(param.requires_grad)

    def test_add_component(self):
        self.fca.add_component()
        self.assertEqual(len(self.fca.parameters_list), 2)

    def test_remove_component(self):
        self.fca.add_component()
        self.assertEqual(len(self.fca.parameters_list), 2)
        self.fca.remove_component(0)
        self.assertEqual(len(self.fca.parameters_list), 1)

    def test_freeze_parameters(self):
        self.fca.add_component()
        self.fca.freeze_parameters(freeze=True)
        for param in self.fca.parameters_list:
            self.assertFalse(param.requires_grad)

    def test_set_fixed(self):
        self.fca.set_fixed(fixed=True)
        self.assertTrue(self.fca.is_fixed)

    def test_reset_fixed_weight(self):
        self.fca.add_component()
        self.fca.reset_fixed_weight()
        self.assertIsNotNone(self.fca.fixed_weight)

    def test_orthogonalize_vectors(self):
        vec = torch.randn(self.size)
        prev_vecs = [torch.randn(self.size) for _ in range(3)]
        self.fca.add_excl_ortho_vectors(prev_vecs)
        self.assertTrue(all_orthogonal(self.fca.orthogonalization_mtx))
        orth_vec = self.fca.orthogonalize_vector(vec, self.fca.orthogonalization_mtx)
        self.assertAlmostEqual(torch.norm(orth_vec, 2).item(), 1.0, places=5)
        for prev_vec in prev_vecs:
            self.assertAlmostEqual(torch.dot(orth_vec, prev_vec).item(), 0.0, places=5)

    def test_update_orthogonalization_mtx(self):
        self.fca.add_component()
        self.fca.update_orthogonalization_mtx(orthogonalize=True)
        for i,orth_vec1 in enumerate(self.fca.orthogonalization_mtx):
            for j,orth_vec2 in enumerate(self.fca.orthogonalization_mtx):
                if i!= j:
                    self.assertAlmostEqual(torch.dot(orth_vec1, orth_vec2).item(), 0.0, places=5)

    def test_reset_fixed_weight(self):
        self.fca.add_component()
        self.fca.reset_fixed_weight()
        self.assertIsNotNone(self.fca.fixed_weight)
        self.assertAlmostEqual(((self.fca.fixed_weight-self.fca.make_matrix())**2).mean().item(), 0, places=5)

    def test_update_parameters(self):
        self.fca.add_component()
        self.fca.update_parameters()
        self.assertEqual(len(self.fca.parameters_list), 2)

    def test_make_matrix(self):
        self.fca.add_component()
        matrix = self.fca.make_matrix()
        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(matrix.shape[1], self.size)

    def test_weight_property(self):
        self.fca.add_component()
        weight = self.fca.weight
        self.assertEqual(weight.shape[1], self.size)

    def test_rank_property(self):
        self.fca.add_component()
        self.assertEqual(self.fca.rank, 2)

    def test_forward(self):
        self.fca.add_component()
        x = torch.randn(5, self.size)
        output = self.fca(x)
        self.assertEqual(output.shape, (5, 2))

    def test_interchange_intervention(self):
        trg = torch.randn(5, self.size)
        src = torch.randn(5, self.size)
        result = self.fca.interchange_intervention(trg, src)
        self.assertEqual(result.shape, trg.shape)

    def test_forward(self):
        self.fca.add_component()
        x = torch.randn(5, self.size)
        output = self.fca(x)
        inv = self.fca(output, inverse=True)
        z = x - inv
        y = self.fca(z)
        self.assertAlmostEqual((y-torch.zeros_like(y)).abs().max().item(), 0.0, places=5)
        

if __name__ == '__main__':
    unittest.main()