
import scipy.spatial
import numpy as np
from .. import MVArray, ConformalLayout
import itertools
import scipy.special
import random
import functools


class GADelaunay(scipy.spatial.Delaunay):
    """
    Uses scipy to produce an hull_dims dimensional Delaunay triangulation of the input
    points

    if hull_dims is not provided it will use the dimension of the input layout
    if hull_dims is provided it will use the first hull_dims basis vectors
    """
    def __init__(self, points, hull_dims=None, furthest_site=False,
                 incremental=False, qhull_options=None):

        # Prep the GA specific stuff
        self.layout = points[0].layout
        if hull_dims is None:
            hull_dims = self.layout.dims
        self.algebra_dims = self.layout.dims
        self.hull_dims = hull_dims
        point_array = np.array([p.value[1:hull_dims+1] for p in points])

        # Run the superclass constructor
        scipy.spatial.Delaunay.__init__(self, point_array,
                                        furthest_site=furthest_site,
                                        incremental=incremental,
                                        qhull_options=qhull_options)
        # Keep the GAversion
        self.GApoints = points

    def conformal_facets(self):
        """
        Returns the list of conformal points in each facet
        """
        if isinstance(self.layout, ConformalLayout):
            return [[self.GApoints[i] for i in s] for s in self.simplices]
        else:
            raise ValueError('Input points do not seem to be from a conformal algebra')


class GAConvexHull(scipy.spatial.ConvexHull):
    """
    Uses scipy to produce an hull_dims dimensional convex hull of the input points
    if hull_dims is not provided it will use the dimension of the input layout
    if hull_dims is provided it will use the first hull_dims basis vectors
    """
    def __init__(self, points, hull_dims=None, incremental=False, qhull_options=None):

        # Prep the GA specific stuff
        self.layout = points[0].layout
        if hull_dims is None:
            hull_dims = self.layout.dims
        self.algebra_dims = self.layout.dims
        self.hull_dims = hull_dims
        point_array = np.array([p.value[1:hull_dims+1] for p in points])

        # Run the superclass constructor
        scipy.spatial.ConvexHull.__init__(self, point_array,
                                          incremental=incremental,
                                          qhull_options=qhull_options)
        # Keep the GAversion
        self.GApoints = points

    def conformal_facets(self):
        """
        Returns the list of conformal points in each facet
        """
        if isinstance(self.layout, ConformalLayout):
            return [[self.GApoints[i] for i in s] for s in self.simplices]
        else:
            raise ValueError('Input points do not seem to be from a conformal algebra')

    def conformal_rounds(self):
        """
        Returns the conformal rounds made of the wedge
        product of the edge simplices
        """
        if isinstance(self.layout, ConformalLayout):
            return [MVArray([self.GApoints[i] for i in s]).op().normal() for s in self.simplices]
        else:
            raise ValueError('Input points do not seem to be from a conformal algebra')

    def conformal_flats(self):
        """
        Returns the conformal flats made of the wedge
        product of the edge simplices with einf
        """
        if isinstance(self.layout, ConformalLayout):
            return [(r^self.layout.einf).normal() for r in self.conformal_rounds()]
        else:
            raise ValueError('Input points do not seem to be from a conformal algebra')


def naiive_subspace_detector(point_list, grade,
                             point_cost_threshold=0.1,
                             flat_flag=True, n_tests=1000):
    """
    Given a point cloud extract objects of the given grade
    This only works for conformal algebras at the moment
    """
    layout = point_list[0].layout
    if not layout.isconformal:
        raise ValueError('The naiive_subspace_detector only operates on conformal algebras at the moment')
    einf = layout.einf

    def sample_objects(point_list, grade, n_tests, flat_flag=True):
        if flat_flag:
            n_sample_objects = grade - 1
        else:
            n_sample_objects = grade

        # How many combos are there
        nchoosek = scipy.special.comb(len(point_list), n_sample_objects)

        # Get a combo generator
        combos = itertools.combinations(point_list, n_sample_objects)

        # Sample from the generator
        n_samples = int(min([n_tests, nchoosek]))
        sampled_objects_list = random.sample(list(combos), n_samples)

        # Wedge the results
        output_list = []
        for i in range(n_samples):
            print(i, flush=True)
            sampled_objects = sampled_objects_list[i]
            wedged_points = functools.reduce(lambda x, y: x ^ y, sampled_objects)
            output_object = wedged_points
            if flat_flag:
                output_object = output_object ^ einf
            final_object = (output_object).normal()
            output_list.append(final_object)
        return output_list

    def associate_points_with_objects(point_list, object_list, point_cost_threshold=0.1):
        # Check how close all the points are to each object
        point_associations = []
        for obj in object_list:
            this_obj_association = []
            for i, p in enumerate(point_list):
                if abs(obj ^ p) < point_cost_threshold:
                    this_obj_association.append(i)
            point_associations.append(this_obj_association)
        return point_associations

    print('SAMPLING', flush=True)
    object_list = sample_objects(point_list, grade, n_tests, flat_flag=flat_flag)

    print('ASSOCIATING', flush=True)
    point_associations = associate_points_with_objects(point_list,
                                                       object_list,
                                                       point_cost_threshold=point_cost_threshold)
    return point_associations, object_list
