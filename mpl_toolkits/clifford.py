from matplotlib.artist import Artist
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from matplotlib.collections import PatchCollection
from matplotlib.tri import Triangulation
from matplotlib.path import Path
from clifford.tools import classify
import clifford
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d, art3d
import math
from trimesh.creation import icosphere
import numpy as np


def _line_style_for_radius(r):
    return '-' if r.imag == 0 else ":"

def plot(ax, cga_objs, **kwargs):
    """
    Plot a multivector or list of multivectors
    """
    if isinstance(cga_objs, clifford.MultiVector):
        cga_objs = [cga_objs]
    if not cga_objs:
        return []

    layout = cga_objs[0].layout
    if not all(cga_obj.layout is layout for cga_obj in cga_objs[1:]):
        raise ValueError("All multivectors must have the same layout")

    if not isinstance(layout, clifford.ConformalLayout):
        raise TypeError("Layout must be conformal")

    if isinstance(ax, Axes3D):
        axis_dims = 3
    else:
        axis_dims = 2
    obj_dims = layout.dims - 2  # for ei, eo
    if obj_dims == axis_dims == 2:
        return _plot_2d(ax, layout, cga_objs, **kwargs)
    elif obj_dims == axis_dims == 3:
        return _plot_3d(ax, layout, cga_objs, **kwargs)
    elif obj_dims == axis_dims:
        raise NotImplementedError("Cannot plot {}-D objects".format(obj_dims))
    elif obj_dims != axis_dims:
        raise TypeError("Objects live in {}-D space, but the axes are {}-D".format(obj_dims, axis_dims))


def _as_point_tuple_2d(p):
    return p[(1,)], p[(2,)]


def _plot_2d(ax, _layout, cga_objs, **kwargs):
    os = [classify.classify(cga_obj) for cga_obj in cga_objs]

    if all(isinstance(o, classify.Point) for o in os):
        x, y = zip(*[_as_point_tuple_2d(point.location) for point in os])
        return ax.plot(x, y, **kwargs)
    elif all(isinstance(o, classify.Tangent[2]) for o in os):
        patches = []
        for tangent in os:
            p = FancyArrowPatch(
                _as_point_tuple_2d(tangent.location),
                _as_point_tuple_2d(tangent.location + tangent.direction.normal()),
                arrowstyle="->", shrinkA=0, shrinkB=0, mutation_scale=10
            )
            ax.add_patch(p)
            patches.append(p)
        return patches
    elif all(isinstance(o, classify.PointPair) for o in os):
        ret = []
        for o in os:
            d = o.direction.normal()
            a = o.location - abs(o.radius) * d
            b = o.location + abs(o.radius) * d
            x, y = zip(*[_as_point_tuple_2d(end) for end in (a, b)])
            ret.append(ax.plot(x, y, linestyle=_line_style_for_radius(o.radius), **kwargs))
        return ret
    elif all(isinstance(o, classify.Line) for o in os):
        return [
            ax.axline(
                _as_point_tuple_2d(line.location - line.direction / 2),
                _as_point_tuple_2d(line.location + line.direction / 2),
                **kwargs
            )
            for line in os
        ]
    elif all(isinstance(o, classify.Circle) for o in os):
        # adjust the color arguments to make sense
        kwargs.setdefault('facecolors', 'none')
        try:
            color = kwargs.pop('color')
        except KeyError:
            raise
        else:
            kwargs['edgecolors'] = color
        c = [
            Circle(_as_point_tuple_2d(circle.location), abs(circle.radius))
            for circle in os
        ]
        linestyles = [_line_style_for_radius(circle.radius) for circle in os]
        col = PatchCollection(c, linestyles=linestyles, **kwargs)
        print(col.get_edgecolors())
        ax.add_collection(col)
        ax._request_autoscale_view()
        return col
    else:
        raise NotImplementedError


def _as_point_tuple_3d(p):
    return p[(1,)], p[(2,)], p[(3,)]


class Arrow3D(FancyArrowPatch):
    def __init__(self, posA, posB, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._posA_3d = posA
        self._posB_3d = posB

    def do_3d_projection(self, renderer):
        xs3d, ys3d, zs3d = zip(self._posA_3d, self._posB_3d)
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return min(zs)


class Circle3D(Patch):
    def __init__(self, center, radius, matrix, **kwargs):
        Patch.__init__(self, **kwargs)
        path = Path.unit_circle()
        self.radius = radius
        self._code3d = path.codes
        print(matrix.shape)
        self._segment3d = center + radius * (path.vertices @ matrix.T)
        self._path2d = Path(np.zeros((0, 2)))

    def get_path(self):
        return self._path2d

    def do_3d_projection(self, renderer):
        s = self._segment3d
        xs, ys, zs = zip(*s)
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs, renderer.M)
        self._path2d = Path(np.column_stack([vxs, vys]), self._code3d)
        return min(vzs)

    # def draw(self, renderer):
    #     self.do_3d_projection()
    #     super().draw(self, renderer)


def _plot_3d(ax, _layout, cga_objs, **kwargs):
    os = [classify.classify(cga_obj) for cga_obj in cga_objs]

    if all(isinstance(o, classify.Point) for o in os):
        coords = zip(*[_as_point_tuple_3d(point.location) for point in os])
        return ax.plot(*coords, **kwargs)
    elif all(isinstance(o, classify.Tangent[2]) for o in os):
        patches = []
        for tangent in os:
            p = Arrow3D(
                _as_point_tuple_3d(tangent.location),
                _as_point_tuple_3d(tangent.location + tangent.direction.normal()),
                arrowstyle="->", shrinkA=0, shrinkB=0, mutation_scale=10
            )
            ax.add_patch(p)
            patches.append(p)
        return patches
    elif all(isinstance(o, classify.PointPair) for o in os):
        ret = []
        for o in os:
            d = o.direction.normal()
            a = o.location - abs(o.radius) * d
            b = o.location + abs(o.radius) * d
            coords = zip(*[_as_point_tuple_3d(end) for end in (a, b)])
            ret.append(ax.plot(*coords, linestyle=_line_style_for_radius(o.radius), **kwargs))
        return ret
    elif all(isinstance(o, classify.Line) for o in os):
        raise NotImplementedError
    elif all(isinstance(o, classify.Circle) for o in os):
        # adjust the color arguments to make sense
        kwargs.setdefault('facecolor', 'none')
        try:
            color = kwargs.pop('color')
        except KeyError:
            raise
        else:
            kwargs['edgecolor'] = color

        e1, e2, e3 = _layout.blades['e1'], _layout.blades['e2'], _layout.blades['e3']
        ret = []
        for circle in os:
            rotor = (circle.direction.normal() * (e1 ^ e2) + 1).normal()
            mat = clifford.linear_operator_as_matrix(lambda x: rotor * x * ~rotor, [e1, e2], [e1, e2, e3])
            p = Circle3D(
                _as_point_tuple_3d(circle.location),
                abs(circle.radius), mat,
                linestyle=_line_style_for_radius(circle.radius),
                **kwargs
            )
            ax.add_patch(p)
            ret.append(p)
        return ret
    elif all(isinstance(o, classify.Sphere) for o in os):
        kwargs.setdefault('alpha', 0.5)
        ret = []
        for o in os:
            tmesh = icosphere(radius=abs(o.radius))
            t = Triangulation(tmesh.vertices[:, 0], tmesh.vertices[:, 1], triangles=tmesh.faces)
            ret.append(ax.plot_trisurf(t, tmesh.vertices[:, 2], **kwargs))
        return ret
    else:
        raise NotImplementedError