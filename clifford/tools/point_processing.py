
import scipy.spatial
import numpy as np
from .. import MVArray


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
        if self.layout.isconformal:
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
        if self.layout.isconformal:
            return [[self.GApoints[i] for i in s] for s in self.simplices]
        else:
            raise ValueError('Input points do not seem to be from a conformal algebra')

    def conformal_rounds(self):
        """
        Returns the conformal rounds made of the wedge
        product of the edge simplices
        """
        if self.layout.isconformal:
            return [MVArray([self.GApoints[i] for i in s]).op().normal() for s in self.simplices]
        else:
            raise ValueError('Input points do not seem to be from a conformal algebra')

    def conformal_flats(self):
        """
        Returns the conformal flats made of the wedge
        product of the edge simplices with einf
        """
        if self.layout.isconformal:
            return [(r^self.layout.einf).normal() for r in self.conformal_rounds()]
        else:
            raise ValueError('Input points do not seem to be from a conformal algebra')
