import unittest
import torch
try:
    from fca import ( orthogonalize_vector, gram_schmidt )
except ImportError:
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from fca import ( orthogonalize_vector, gram_schmidt )

class TestGramSchmidt(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.vec = torch.randn(self.size)
        self.prev_vecs = [torch.randn(self.size) for _ in range(3)]
    
    def test_orthogonalize_first_vector(self):
        ground_truth = self.vec/torch.norm(self.vec, 2)
        ortho_vec = orthogonalize_vector(self.vec, [])
        self.assertAlmostEqual(
            ((ground_truth-ortho_vec)**2).mean().item(),
            0, places=5)

    def test_orthogonalize_with_previous_vectors(self):
        orth_vec = orthogonalize_vector(self.vec, self.prev_vecs)
        ortho_vecs = []
        for prev_vec in self.prev_vecs:
            ortho_vec = orthogonalize_vector(prev_vec, ortho_vecs)
            ortho_vecs.append(ortho_vec)
        for i,ortho_vec in enumerate(ortho_vecs):
            self.assertAlmostEqual(torch.norm(ortho_vec, 2).item(), 1.0, places=5)
            for j,comp_vec in enumerate(ortho_vecs):
                if i != j:
                    self.assertAlmostEqual(torch.dot(ortho_vec, comp_vec).item(), 0.0, places=5)
    
    def test_gram_schmidt(self):
        ortho_vecs = gram_schmidt(self.prev_vecs)
        for i, ortho_vec in enumerate(ortho_vecs):
            self.assertAlmostEqual(torch.norm(ortho_vec, 2).item(), 1.0, places=5)
            for j, comp_vec in enumerate(ortho_vecs):
                if i != j:
                    self.assertAlmostEqual(torch.dot(ortho_vec, comp_vec).item(), 0.0, places=5)
        new_vec = orthogonalize_vector(self.vec, ortho_vecs)
        self.assertAlmostEqual(torch.norm(new_vec, 2).item(), 1.0, places=5)
        for ortho_vec in ortho_vecs:
            self.assertAlmostEqual(torch.dot(new_vec, ortho_vec).item(), 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
