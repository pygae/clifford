from clifford import MultiVector, Cl, randomMV
from numpy.random import uniform
import unittest

# Put the names of the blades into the module namespace.
# This is a bit of a hack, but it's quite convenient.
#import __main__
#for name, value in blades.items():
#    setattr(__main__, name, value)

class CliffordTests(unittest.TestCase):

    def setUp(self):
        self.algebras = [Cl(i) for i in [3, 4, 5, 6]]

    def test_inverse(self):
        for layout, blades in self.algebras:
            a = 1. + blades['e0']
            self.assertRaises(ValueError, lambda x: 1/x, a)
            for i in range(10):
                a = randomMV(layout, grades=[0, 1])
                denominator = float(a(1)**2-a(0)**2)
                if abs(denominator) > 1.e-5:
                    a_inv = (-a(0)/denominator) + ((1./denominator) & a(1))
                    self.assert_(abs((a & a_inv)-1.) < 1.e-11)
                    self.assert_(abs((a_inv & a)-1.) < 1.e-11)
                    self.assert_(abs(a_inv - 1./a) < 1.e-11)
    
    def test_exp(self):
        layout,blades = self.algebras[0]:
        R = e**blades['e01']
        e0 = blades['e0']
        R& e0 &~R
        
if __name__ == '__main__':
    unittest.main()

