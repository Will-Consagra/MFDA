from .domain import Domain

import numpy as np
from mshr import Circle, Sphere, generate_mesh
from dolfin import Point 

class pSphere(Domain):
	"""
	Spherical domain

	Attributes:
	
	Notes: 
		- Which coordinate system to use? See geomstats package for inspiration. For now, just use (x, y, z) coords and peform 
		a check that any input domain points are 'close enough'.
	"""

	def __init__(self, p=2, x0=(0, 0, 0), radius=1, resolution=10):
		if not isinstance(p, int) or p < 1:
			raise ValueError
		if p > 2:
			raise NotImplementedError("Hypersphere support is currently not available")
		self.p = p 
		self.origin = x0
		self.radius = 1
		self.resolution = resolution
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

	def in_domain(self, X, epsilon=0.1):
		"""
		Argurments:
			X: N x p dimensional numpy array defining point clouds
			epsilon: tolerance for deviation from surface of S_p
		Returns: 
			List of booleans indicating whether the row is in S_p 
		"""
		sample_radius = np.sqrt(np.sum(X**2, axis=1))
		return np.abs(sample_radius - self.radius) > epsilon

	def _inner_product(self, X1, X2):
		raise NotImplementedError 

	def _distance(self, X1, X2):
		raise NotImplementedError

	def create_mesh(self):
		"""
		Arguments:
			resolution: int defining the resolution of the resulting tessellation
		"""
		if self.p == 1:
			domain = Circle(Point(*self.origin), self.radius)
		elif self.p == 2:
			domain = Sphere(Point(*self.origin), self.radius)
		else:
			raise NotImplementedError("p > 2 not yet implemented!")
		self.mesh = generate_mesh(domain, self.resolution)
		return self