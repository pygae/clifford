
from __future__ import print_function
from clifford import grade_obj
from clifford.g3c import *
from clifford.tools.g3c import interpret_multivector_as_object
from clifford.io import read_ga_file
from clifford import MVArray
import json


def interpolate_colors(alpha, rgb_a, rgb_b):
    rgb_c = (1-alpha)*rgb_a + alpha*rgb_b
    return 'rgb(' + str(int(rgb_c[0])) + ',' + str(int(rgb_c[1])) + ',' + str(int(rgb_c[2])) + ')'


class Sphere():
    def __init__(self, mv, color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawSphere('+str(self.mv)+','+self.color+');'
        return return_string


class Plane():
    def __init__(self, mv, color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawPlane('+str(self.mv)+','+self.color+');'
        return return_string


class Line():
    def __init__(self, mv, color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawLine('+str(self.mv)+','+self.color+');'
        return return_string


class Circle():
    def __init__(self, mv, color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawCircle('+str(self.mv)+','+self.color+');'
        return return_string


class PointPair():
    def __init__(self, mv, color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawPointPair('+str(self.mv)+','+self.color+');'
        return return_string


class EucPoint():
    def __init__(self, mv, color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawEucPoint('+str(self.mv)+','+self.color+');'
        return return_string


class ConfPoint():
    def __init__(self, mv, color):
        self.mv = mv
        self.color = color

    def gen_string(self):
        return_string = 'DrawConfPoint('+str(self.mv)+','+self.color+');'
        return return_string


class GAScene():

    def __init__(self):
        self.objects = []

    def add(self, object_added):
        self.objects.append(object_added)

    def concat(self, other):
        if isinstance(other, GAScene):
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

    def add_euc_point(self, mv, color='rgb(0,0,0)'):
        if grade_obj(mv) != 1:
            raise ValueError('Input is not a euc_point')
        self.add(EucPoint(mv, color))

    def add_conf_point(self, mv, color='rgb(0,0,0)'):
        if grade_obj(mv) != 1:
            raise ValueError('Input is not a conf_point')
        self.add(ConfPoint(mv, color))

    def add_circle(self, mv, color='rgb(0,0,0)'):
        if grade_obj(mv) != 3:
            raise ValueError('Input is not a circle')
        self.add(Circle(mv, color))

    def add_line(self, mv, color='rgb(0,0,0)'):
        if grade_obj(mv) != 3:
            raise ValueError('Input is not a line')
        self.add(Line(mv, color))

    def add_plane(self, mv, color='rgb(0,0,0)'):
        if grade_obj(mv) != 4:
            raise ValueError('Input is not a plane')
        self.add(Plane(mv, color))

    def add_sphere(self, mv, color='rgb(0,0,0)'):
        if grade_obj(mv) != 4:
            raise ValueError('Input is not a sphere')
        self.add(Sphere(mv, color))

    def add_point_pair(self, mv, color='rgb(0,0,0)'):
        if grade_obj(mv) != 2:
            raise ValueError('Input is not a point_pair')
        self.add(PointPair(mv, color))

    def add_object(self, mv, mv_type='interp', color='rgb(0,0,0)'):
        if mv_type == 'line':
            self.add_line(mv, color)
        elif mv_type == 'conf_point':
            self.add_conf_point(mv, color)
        elif mv_type == 'euc_point':
            self.add_euc_point(mv, color)
        elif mv_type == 'point_pair':
            self.add_point_pair(mv, color)
        elif mv_type == 'circle':
            self.add_circle(mv, color)
        elif mv_type == 'plane':
            self.add_plane(mv, color)
        elif mv_type == 'sphere':
            self.add_sphere(mv, color)
        elif mv_type == 'interp':
            """
            -1 -> not a blade
            0 -> a 1 vector but not a point
            1 -> a euclidean point
            2 -> a conformal point
            3 -> a point pair
            4 -> a circle
            5 -> a line
            6 -> a sphere
            7 -> a plane
            """
            return_val = interpret_multivector_as_object(mv)
            if return_val > 0:
                if return_val == 1:
                    self.add_euc_point(mv, color=color)
                elif return_val == 2:
                    self.add_conf_point(mv, color=color)
                elif return_val == 3:
                    self.add_point_pair(mv, color=color)
                elif return_val == 4:
                    self.add_circle(mv, color=color)
                elif return_val == 5:
                    self.add_line(mv, color=color)
                elif return_val == 6:
                    self.add_sphere(mv, color=color)
                elif return_val == 7:
                    self.add_plane(mv, color=color)
            else:
                raise ValueError('Input object cannot be interpretted as an object.')
        else:
            raise ValueError(str(mv_type) + ' is not a valid mv_type. You must specify a valid mv_type.')

    def add_object_array(self, mv_array, mv_type='interp', color='rgb(0,0,0)'):
        for mv in mv_array:
            self.add_object(mv, mv_type, color=color)

    def save_to_file(self, filename):
        with open(filename, 'w') as fobj:
            print(self, file=fobj)


class GanjaScene:

    def __init__(self):
        self.internal_list = []

    def add_object(self, mv, color=int('AA000000', 16), label=None):
        if label is not None:
            if not isinstance(label, str):
                raise ValueError('Labels must be strings')
            self.internal_list.append(label)
        self.internal_list.append(color)
        self.internal_list.append([i for i in mv.value])

    def add_objects(self, mv_list, color=int('AA000000', 16), label=None):
        if label is not None:
            if not isinstance(label, str):
                raise ValueError('Labels must be strings')
            self.internal_list.append(label)
        self.internal_list.append(color)
        for mv in mv_list:
            self.internal_list.append([i for i in mv.value])

    def __repr__(self):
        return json.dumps(self.internal_list)

    def save_to_file(self, filename):
        with open(filename, 'w') as fobj:
            print(self, file=fobj)


def draw_objects_ganja(objects, color=int('AA000000', 16), print_scene=True):
    """
    Takes a list of multivectors or a .ga file name and draws the multivectors
    By default attempts to interpret the type of object unless a mv_type is specified
    """
    if isinstance(objects, str):
        data_array, metric, basis_names, support = read_ga_file(objects)
        mv_list = [layout.MultiVector(data_array[i, :]) for i in range(data_array.shape[0])]
        sc = GanjaScene()
        sc.add_objects(mv_list, color=color)
        if print_scene:
            print(sc)
        return sc
    elif isinstance(objects, list) or isinstance(objects, MVArray):
        sc = GanjaScene()
        sc.add_objects(objects, color=color)
        if print_scene:
            print(sc)
        return sc
    else:
        raise ValueError('The input is not a string or a list of objects')


def draw_objects(objects, mv_type='interp', color='rgb(0,0,0)', print_scene=True):
    """
    Takes a list of multivectors or a .ga file name and draws the multivectors
    By default attempts to interpret the type of object unless a mv_type is specified
    """
    if isinstance(objects, str):
        data_array, metric, basis_names, support = read_ga_file(objects)
        mv_list = [layout.MultiVector(data_array[i, :]) for i in range(data_array.shape[0])]
        sc = GAScene()
        sc.add_object_array(mv_list, mv_type, color=color)
        if print_scene:
            print(sc)
        return sc
    elif isinstance(objects, list) or isinstance(objects, MVArray):
        sc = GAScene()
        sc.add_object_array(objects, mv_type, color=color)
        if print_scene:
            print(sc)
        return sc
    else:
        raise ValueError('The input is not a string or a list of objects')


class GAAnimation():
    def __init__(self):
        self.scenes = []

    def add_scene(self, scene):
        self.scenes.append(scene)

    def __repr__(self):
        s = ''
        for obj in self.scenes:
            s += str(obj).replace("\n", "#") + '\n'
        s = s[:-1]
        return s

    def save_to_file(self, filename):
        with open(filename, 'w') as fobj:
            print(self, file=fobj)
