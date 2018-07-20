from functools import reduce
from . import conformalize, op,gp, MultiVector,Cl
from numpy import zeros,e
import math
    
class Round(object):
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
            nulls = map(self.cga.null_vector,args)
            self.mv = reduce(op,nulls) 
        
        self.mv = self.mv.normal()
        
    def from_center_radius(self,center, radius):
        center = self.cga.null_vector(center)
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
            mv = 1 - self.cga.lower_vector()*self.cga.einf/2.
        
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
        if isinstance( layout_orig,int):
            layout_orig = Cl(layout_orig)
        self.layout_orig = layout_orig
        self.layout, self.blades, stuff = conformalize(layout_orig)
        self.__dict__.update(stuff)
    
    
    ## Objects
    def lower_vector(self):
        '''
        random vector in the lower(original) space
        '''
        return self.I_base.project(self.randomV())
    
    def null_vector(self,x=None):
        '''
        generates random null vector if x is None, or
        returns a null vector from base vector x, if  x^self.I_base ==0 
        returns x, 
        
        a null vector will lay on the horisphere
        '''
        if x is None:
            return self.up(self.lower_vector())
        else:
            if x^self.I_base ==0:
                return self.up(x)
            return x
    
    def straight_up(self, x):
        '''
        place a vector from layout_orig into this CGA, without up()
        '''
        return self.I_base.project(self.up(x))
    
    
    def round(self, *args):
        return Round(self,*args) 
    
    ##  Operators
    def translation(self, *args):
        return Translation(self, *args)

    
