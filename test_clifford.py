from clifford import MultiVector, Cl, randomMV,eps, pretty,gp
from clifford.tools import orthoFrames2Verser

from numpy.random import uniform
from numpy import  exp, float64,e
import unittest

from nose.plugins.skip import SkipTest

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
                    a_inv = (-a(0)/denominator) + ((1./denominator) * a(1))
                    self.assert_(abs((a * a_inv)-1.) < 1.e-11)
                    self.assert_(abs((a_inv * a)-1.) < 1.e-11)
                    self.assert_(abs(a_inv - 1./a) < 1.e-11)


    
    def test_exp(self):
        
        layout, blades = self.algebras[0]
        R = exp(blades['e01'])
        e0 = blades['e0']
        R*e0*~R
        
    def test_add_float64(self):
        
        layout, blades = self.algebras[0]
        e0 = blades['e0']
        
        
        float64(1)+e0
        self.assertEqual(1+e0,float64(1)+e0)


    
    

@SkipTest
class ToolsTests(unittest.TestCase):
    def testOrthoFrames2Verser(self):
        for p,q in [[4,0],[3,1]]:
            #p,q =4,0
            N=p+q
            eps(1e-6)
            layout, blades = Cl(p,q)
            
            # create frame 
            A = layout.randomV(n=N, normed=True)
            # create Rotor
            R = layout.randomRotor()
            # create rotated frame
            B =  [R*a*~R for a in A]
            
            # find verser from both frames
            R_found = orthoFrames2Verser(A,B)
            
            pretty()
        
            print p,q
            print R
            print R_found
            #raise ValueError
            # Rotor is determiend correctly, within a sign
            self.assertTrue(R==R_found  or R==-R_found)
            
            # Determined Verser implements desired transformation  
            self.assertTrue([R_found*a*~R_found for a in A] ==B)




if __name__ == '__main__':
    unittest.main()

