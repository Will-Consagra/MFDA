from .basis import FBasis

import numpy as np

from dolfin import *

class FE(FBasis):
	"""
	Piecewise linear basis on triangulation 

	Attributes: 

	Notes: We take inspiration from the sklearn fit - predict api. An object of class MPF can be instantiated but must be 
		   "fit" before using methods such as evaluate and so on. Must implement a similiar method to check_is_fitted in sklearn.
	"""

	def __init__(self, domain, metric=None, seeds=None):
		## perform required check/validations 
		## instantiate attributes 
		self.domain = domain
		self.metric = metric
		self.seeds = seeds

	def construct_function_space(self):
		"""
		Construct the function space.
		Eventually we want to allow this to work purely from training data (X, Y).
		For now, we adapt to the predefined metric and the (X, Y) pair is ignored.
		"""
		if self.metric:
			## perform anisotropic tessellation 
			raise NotImplementedError
		elif self.seeds is not None:
			## construct a Delaunay tessellation of the seeds in self.seeds
			raise NotImplementedError
		else:
			self.domain.create_mesh()
		self.function_space = FunctionSpace(self.domain.mesh, "CG", 1)
		self.K = self.function_space.dim()

	def _evaluate(self, X):
		"""
		Compute each of the K basis functions at each of the points in X; Nxp numpy array 
		Arguments:
			X: domain points, Nxp array 
		Returns:
			Phi: Basis function evaluates at each point in X, NxK numpy array 
		Notes:
			- Need to make this work for both panar and curved domains, e.g. curved domains may not have domain points 
			falling exactly within the elements.
		"""
		N = X.shape[0]
		mesh = self.domain.mesh 
		V = self.function_space 
		Phi = np.zeros((N, self.K))
		tree = mesh.bounding_box_tree()
		dofmap = V.dofmap()
		elements = V.element()
		for j in range(N):
			x = X[j, :]
			#cell_id = tree.compute_first_entity_collision(Point(*x)) 
			cell_id, dist = tree.compute_closest_entity(Point(*x))
			cell = Cell(mesh, cell_id)
			cell_global_dofs = dofmap.cell_dofs(cell_id)
			coordinate_dofs = cell.get_vertex_coordinates()
			cell_orientation = cell.orientation()
			for local_dof in range(len(cell_global_dofs)):
				Phi[j, cell_global_dofs[local_dof]] = elements.evaluate_basis(local_dof, x, coordinate_dofs, cell_orientation)
		return Phi

	def _gradient(self):
		raise NotImplementedError

	def _inner_product_matrix(self, other=None):
		"""
		Note: Eventaully want to use sparse matrices 
		"""
		u = TrialFunction(self.function_space)
		v = TestFunction(self.function_space)
		J = assemble(u*v*dx)
		return J.array()

	def _weak_roughness_matrix(self):
		u = TrialFunction(self.function_space)
		v = TestFunction(self.function_space)
		R = assemble(inner(grad(u),grad(v))*dx)
		return R.array()

	def _roughness_matrix(self):
		return self._weak_roughness_matrix()
