from .domain import Domain 

from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh

import numpy as np
import types 
import itertools
from scipy.spatial import Delaunay

class Hypercube(Domain):
	"""
	A (unit) hypercube domain.

	Attributes: 

	Notes: 
	"""

	def __init__(self, p, marginal_seeds=None):
		if not isinstance(p, int) or p < 1:
			raise ValueError
		self.p = p 
		self.marginal_seeds = marginal_seeds
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

	def create_mesh(self):
		"""
		Arguments:
			marginal_seeds: tuple of length p specifying the number of marginal seeds in the tessellation
		"""
		if self.p == 1:
			self.mesh = UnitIntervalMesh(*self.marginal_seeds)
		elif self.p == 2:
			self.mesh = UnitSquareMesh(*self.marginal_seeds)
		elif self.p == 3:
			self.mesh = UnitCubeMesh(*self.marginal_seeds)
		else: 
			seeds = np.array(list(itertools.product(*[list(range(self.marginal_seeds[i])) for i in range(self.p)])))
			Dtess = Delaunay(seeds) 
			self.mesh = Dtess ## Note: this object is of different type than the other meshes!!
		return self

	