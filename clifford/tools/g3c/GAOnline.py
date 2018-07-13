
from __future__ import print_function


class Sphere():
    def __init__(self,mv,color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawSphere('+str(self.mv)+','+self.color+');'
        return return_string


class Plane():
    def __init__(self,mv,color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawPlane('+str(self.mv)+','+self.color+');'
        return return_string


class Line():
    def __init__(self,mv,color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawLine('+str(self.mv)+','+self.color+');'
        return return_string


class Circle():
    def __init__(self,mv,color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawCircle('+str(self.mv)+','+self.color+');'
        return return_string


class PointPair():
    def __init__(self,mv,color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawPointPair('+str(self.mv)+','+self.color+');'
        return return_string


class EucPoint():
    def __init__(self,mv,color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawEucPoint('+str(self.mv)+','+self.color+');'
        return return_string


class GAScene():

    def __init__(self):
        self.objects = []

    def add(self,object_added):
        self.objects.append(object_added)

    def concat(self,other):
        if isinstance(other,GAScene):
            for obj in other.objects:
                self.add(obj)
        else:
            raise ValueError('Other object must be a GAScene')

    def clear(self):
        self.objects = []

    def __repr__(self):
        s = ''
        for obj in self.objects:
            s += obj.gen_string() + '\n'
        s = s[:-1]
        return s

    def add_euc_point(self,mv,color='rgb(0,0,0)'):
        self.add(EucPoint(mv,color))

    def add_circle(self,mv,color='rgb(0,0,0)'):
        self.add(Circle(mv,color))

    def add_line(self,mv,color='rgb(0,0,0)'):
        self.add(Line(mv,color))

    def add_plane(self,mv,color='rgb(0,0,0)'):
        self.add(Plane(mv,color))

    def add_sphere(self,mv,color='rgb(0,0,0)'):
        self.add(Sphere(mv,color))

    def add_point_pair(self,mv,color='rgb(0,0,0)'):
        self.add(PointPair(mv,color))

    def add_object(self,mv,mv_type,color='rgb(0,0,0)'):
        if mv_type == 'line':
            self.add_line(mv, color)
        elif mv_type == 'point_pair':
            self.add_point_pair(mv, color)
        elif mv_type == 'circle':
            self.add_circle(mv, color)
        elif mv_type == 'plane':
            self.add_plane(mv, color)
        elif mv_type == 'sphere':
            self.add_sphere(mv, color)
        else:
            raise ValueError(str(mv_type) + ' is not a valid mv_type. You must specify a valid mv_type.')

    def save_to_file(self,filename):
        with open(filename,'w') as fobj:
            print(self,file=fobj)

class GAAnimation():
    def __init__(self):
        self.scenes = []

    def add_scene(self,scene):
        self.scenes.append(scene)

    def __repr__(self):
        s = ''
        for obj in self.scenes:
            s += str(obj).replace("\n", "#") + '\n'
        s = s[:-1]
        return s

    def save_to_file(self,filename):
        with open(filename,'w') as fobj:
            print(self,file=fobj)