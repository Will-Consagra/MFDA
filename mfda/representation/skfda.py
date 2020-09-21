from .basis import FBasis
import sys 
sys.path.append("..")
from domains.hypercube import Hypercube 
from skfda.representation.basis import BSpline, FDataBasis
import numpy as np 

class bspline(FBasis):
	"""
	Thin wrapper for Bspline basis class from sci-kit fda package
	
	Arguments:
		bspline_args: arguments to BSpline class 
	Notes:
		this is currently more of a workaround than an acutal integration 
	"""

	def __init__(self, bspline_args):
		self.domain = Hypercube(1)
		self.skfda_basis = BSpline(domain_range=(0,1), **bspline_args)
		self.K = self.skfda_basis.n_basis

	def _evaluate(self, X):
		"""
		Compute each of the K basis functions at each of the points in X
		Arguments:
			X: domain points, length N array 
		Returns:
			Phi: Basis function evaluates at each point in X, NxK numpy array 
		"""
		return self.skfda_basis.evaluate(X).T


	def _gradient(self):
		raise NotImplementedError

	def _inner_product_matrix(self, other=None):
		"""
		Note: Eventaully want to use sparse matrices 
		"""
		return self.skfda_basis.gram_matrix()

	def _roughness_matrix(self):
		return self.skfda_basis.penalty(derivative_degree=2)

	def cross_roughness_matrix(self):
		basis_fdata = FDataBasis(self.skfda_basis, np.eye(self.K))
		basis_fdata_deriv = basis_fdata.derivative(2)
		return basis_fdata.inner_product(basis_fdata_deriv)