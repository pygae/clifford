
from __future__ import print_function
from clifford import grade_obj

def interpolate_colors(alpha, rgb_a, rgb_b):
    rgb_c = (1-alpha)*rgb_a + alpha*rgb_b
    return 'rgb(' + str(int(rgb_c[0])) + ',' + str(int(rgb_c[1])) + ',' + str(int(rgb_c[2])) + ')'


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
        if grade_obj(mv) != 1:
            raise ValueError('Input is not a euc_point')
        self.add(EucPoint(mv,color))

    def add_circle(self,mv,color='rgb(0,0,0)'):
        if grade_obj(mv) != 3:
            raise ValueError('Input is not a circle')
        self.add(Circle(mv,color))

    def add_line(self,mv,color='rgb(0,0,0)'):
        if grade_obj(mv) != 3:
            raise ValueError('Input is not a line')
        self.add(Line(mv,color))

    def add_plane(self,mv,color='rgb(0,0,0)'):
        if grade_obj(mv) != 4:
            raise ValueError('Input is not a plane')
        self.add(Plane(mv,color))

    def add_sphere(self,mv,color='rgb(0,0,0)'):
        if grade_obj(mv) != 4:
            raise ValueError('Input is not a sphere')
        self.add(Sphere(mv,color))

    def add_point_pair(self,mv,color='rgb(0,0,0)'):
        if grade_obj(mv) != 2:
            raise ValueError('Input is not a point_pair')
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

    def add_object_array(self,mv_array,mv_type,color='rgb(0,0,0)'):
        for mv in mv_array:
            self.add_object(mv, mv_type, color=color)

    def save_to_file(self,filename):
        with open(filename,'w') as fobj:
            print(self,file=fobj)


def draw_objects(mv_array, mv_type, color='rgb(0,0,0)'):
    sc = GAScene()
    sc.add_object_array(mv_array, mv_type, color=color)
    print(sc)


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