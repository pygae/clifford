from functools import reduce
from . import conformalize, op,gp, MultiVector
from numpy import zeros,e
import math
    
class Sphere(object):
    # could inherent some generic CGAObject class
    def __init__(self,cga, *args):
        self.cga = cga
        self.layout = cga.layout # note: self.layout is the cga layout
        
        if len(args) ==0:    
            # generate highest dimension sphere
            points = [self.cga.lower_vector() for k in range(self.layout.dims-1)]
            points = map(self.cga.up, points)
            self.mv = reduce(op,points) 
        
        elif len(args)==1:
            if isinstance(args[0], MultiVector):
                self.mv = args[0]
            if isinstance(args[0], int):
                dim = args[0]
                points = [self.cga.lower_vector() for k in range(dim+2)]
                points = map(self.cga.up, points)
                self.mv = reduce(op,points)
                
                
        else:
            points = args
            if points[0].layout == self.cga.layout_orig:
                points = map(self.cga.up,points)
            
            # verify points are points
            self.mv = reduce(op,points) 
        
        self.mv = self.mv.normal()
        
    def from_center_radius(self,center, radius):
        if center.layout == self.cga.layout_orig:
            center = self.cga.up(center)
        self.mv = center - .5*radius**2*einf
    
    @property
    def dim(self):
        return self.layout.dims 
    
    @property
    def center(self):
        return self.mv * self.cga.einf*self.mv
    
    @property
    def center_down(self):
        return self.cga.down(self.center)
    @property
    def radius(self):
        dual_sphere = self.dual 
        dual_sphere / (-dual_sphere | self.cga.einf)
        return math.sqrt(abs(dual_sphere * dual_sphere))

    @property 
    def dual(self):
        return self.mv* self.layout.pseudoScalar
        
class CGAOperator(object):
    def __call__(self, other):
        if isinstance(other, MultiVector):
            return self.mv*other*~self.mv
        else:
            klass = other.__class__
            return klass(self.cga, self.mv*other.mv*~self.mv)
                
class Translation(CGAOperator):
    def __init__(self,cga, *args):
        self.cga = cga
        self.layout = cga.layout
        
        if len(args) ==0:    
            # generate generator!
            mv = 1 - self.cga.lower_vector_hi()*self.cga.einf/2.
        
        elif len(args)==1:
            arg = args[0]
            if isinstance(arg, MultiVector):
                if arg.grades()==[1]:
                    # we have vector 
                    mv = 1 - self.cga.straight_up(arg)*self.cga.einf/2.     
        else:
            raise ValueError('bad input')
            
        self.mv = mv
    
class CGA(object):
    def __init__(self, layout_orig):
        '''
        Conformal Geometric Algebra 
        
        conformalizes the layout_orig,  and provides several
        methods and for objects/operators
        '''
        self.layout_orig = layout_orig
        self.layout, self.blafdes, stuff = conformalize(layout_orig)
        self.__dict__.update(stuff)
    
    ## Objects
    def lower_vector(self):
        '''
        random vector in the lower(original) space
        '''
        return self.layout_orig.randomV()
    
    def null_vector(self):
        '''
        up projected random vector from original space 
        will lay on horisphere
        '''
        return self.up(self.lower_vector())
    
    def straight_up(self, x):
        '''
        copy of lower vector x in this cga, (without  added dims)
        '''
        return self.I_ga.project(self.up(x))
    
    def lower_vector_hi(self):
        '''
        a random lower vector straight_up projected into this cga
        '''
        return self.straight_up(self.layout.randomV())
    
    
    def sphere(self, *args):
        
        return Sphere(self,*args) 
    
    ##  Operators
    def translation(self, *args):
        return Translation(self, *args)
