from abc import ABC, abstractmethod

class FBasis(ABC):
	"""
	Defines the template for a function basis object
	
	Attributes: 
		domain: Domain object 
		K: Rank (number of basis functions)

	"""

	@abstractmethod 
	def __init__(self, domain):
		## Perform parameter checks and set arguments as corresponding class attributes
		self.domain = domain
		
	def __repr__(self):
		"""Representation of an FBasis object"""
		return("")

	@abstractmethod
	def _evaluate(self, X):
		""" Evaluate basis function on point cloud X (Nxp) """
		pass

	@abstractmethod
	def _gradient(self):
		""" 
		Return the gradient of the basis object
		"""
		pass

	@abstractmethod
	def _inner_product_matrix(self, other=None):
		"""
		Returns the inner product matrix of itself with other

		Parameters:

		Returns:
		"""
		pass 

	@abstractmethod
	def _roughness_matrix(self):
		"""
		Returns roughness penality matrix (pairwise inner product of Laplacian of basis functions)
		"""
		pass 

	def evaluate(self, X):
		"""
		Performs the appropriate checks before invoking _evaluate
		
		Parameters: 
			X: points in the domain, point cloud like 
		Returns: 
		"""
		return self._evaluate(X) 

	def gram_matrix(self):
		"""
		Compute the pairwise inner product matrix of the basis system

		Parameters: None

		Returns: 
		"""
		return self._inner_product_matrix()

	def roughness_matrix(self):
		return self._roughness_matrix()
