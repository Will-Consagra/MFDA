from .domain import Domain 

import numpy as np
import types 
from scipy.spatial import Delaunay

class Hypercube(Domain):
	"""
	A (unit) hypercube domain.

	Attributes: 

	Notes: 
	"""

	def __init__(self, p):
		if not isinstance(p, int) or p < 1:
			raise ValueError
		self.p = p 
		self._mesh = None

	@property
	def dim(self):
		return self.p
	
	@property
	def mesh(self):
		return self._mesh

	@mesh.setter
	def mesh(self, value):
		self._mesh = value

	@mesh.deleter
	def mesh(self):
		del self._mesh
	
	def in_domain(self, X):
		"""
		Check if the data X is within the unit hypercube.

		X: list of either 1) N_i x p dimensional numpy array defining point clouds or 2) numpy arrays of n_i marginal grid coordinates 

		ToDo:
			1) Should be able to deal with an ndarry as well as a list 
			2) Should add a message to all of the value errors
		"""
		## Should call utils.validation.infer_X_data on X
		## Check the data format is accepatable 
		if not isinstance(X, list):
			raise ValueError
		if len(X[0].shape) == 1: ## Data is arranged on a regular p-dimenional grid
			assert len(X) == self.p, "Inconsistent dimensions!!"
			lower_bounds = np.array([np.min(X[i]) for i in range(self.p)])
			upper_bounds = np.array([np.max(X[i]) for i in range(self.p)])
		else: ## Data is a list of point clouds 
			assert all([X[i].shape[1] == self.p for i in range(len(X))]), "Inconsistent dimensions!!"
			X_stack = np.concatenate(X, axis=0)
			lower_bounds = np.min(X_stack, 0)
			upper_bounds = np.max(X_stack, 0)
		if not (np.all(lower_bounds >= 0) and (np.all(upper_bounds) <= 1)):
			return False 
		return True 

	def _inner_product(self, X1, X2):
		return X1 @ X2.T

	def _distance(self, X1, X2):
		Diff = X1 - X2 
		return Diff @ Diff.T

	def aniso_tessellation(self, seeds, metric=None):
		"""
		Perform a tessellation of the unit hypercube

		Parameters: 
				   seeds: Mxp array of vertices for the tessellation 
				   metric: None or one for performing anisotropic tessellation 

		Notes: - This function will call some function in the mesh_adaptation.py file 
				to perform the heavy lifting.
				- Should add option of whether we want an anisotropic mesh over a fixed set of seeds or to re-sample the seeds based on 
					the local metric field defined by metric 
		"""
		if not self.in_domain([seeds]):
			raise ValueError

		Dtess = Delaunay(seeds)

		if isinstance(metric, types.FunctionType):
			mesh = {}
			if self.p == 2:
				mesh["xy"] = Dtess.points 
				mesh["Triangles"] = Dtess.simplices
			elif self.p == 3:
				mesh["xyz"] = Dtess.points 
				mesh["Tetrahedra"] = Dtess.simplices
			else:
				raise ValueError("Anisotropic meshing not yet implemented for p > 3") 
			## Call anisotropic meshing functions in mesh_adaptation.py 

		self.mesh = mesh

	def marginal_domains(self):
		"""
		Eventually this will be inherited from ProductManfold class 
		"""
		return [(0,1)]*self.p