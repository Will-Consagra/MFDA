from .basis import FBasis 
import sys 
sys.path.append("..")
from domains.sphere import pSphere 
import numpy as np 

from dipy.reconst.shm import real_sym_sh_basis
## Utility Functions 
def cart2sphere(x):
	r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
	theta = np.arctan2(x[:,1], x[:,0])
	phi = np.arccos(x[:,2]/r)
	return np.column_stack([theta, phi])

class RealSymSpH(FBasis):
	"""
	Implements a real, symmetric spherical harmonic basis system in the FBasis interface 

	Arguments:
		sh_order: order of harmonic 
	Notes: 
	"""

	def __init__(self, sh_order):
		self.domain = pSphere()
		self.K = int((sh_order+1)*(sh_order+2)/2)
		self.sh_order = sh_order

	def _evaluate(self, X):
		"""
		Compute each of the K basis functions at each of the points in X
		Arguments:
			X: domain points, length N array 
		Returns:
			Phi: Basis function evaluates at each point in X, NxK numpy array 
		"""
		X_spherical = cart2sphere(X)
		theta = X_spherical[:,0]; phi = X_spherical[:,1]
		B, m, n = real_sym_sh_basis(self.sh_order, phi, theta)
		return B

	def _gradient(self):
		raise NotImplementedError

	def _inner_product_matrix(self, other=None):
		"""
		Note: Eventaully want to use sparse matrices 
		"""
		return np.eye(self.K)

	def _roughness_matrix(self):
		return NotImplementedError