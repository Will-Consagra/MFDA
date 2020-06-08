from .linear import LinearSmoother

import numpy as np 

class RidgeSmoother(LinearSmoother):
	"""
	Implements a ridge-based linear smoothing operator
	Attributes:
		   basis: inherits from FBasis object 
		   smoothing_parameter: float >= 0 specifying the ridge penalty
		   regularization: specifies the type of penalty matrix to use, for now only roughness penalty in implemented 
	Notes:
		  - Extend the types of penalty matrices that can be used. 
		  - Allow a weight matrix for heteroskedasticity 
	"""
	def __init__(self, basis, smoothing_parameter, regularization="roughness_matrix"):
		self.basis = basis
		self.smoothing_parameter = smoothing_parameter
		self.regularization = regularization

	def _hat_matrix(self, X):
		"""
		Compute hat matrix for ridge smoothing.
		Arguments: 
			X: domain points for which the observations to be smoothed have been calculated. 
		"""
		Phi = self.basis.evaluate(X)
		R = getattr(self.basis, self.regularization)()
		self.H = Phi @ np.linalg.inv(Phi.T @ Phi + self.smoothing_parameter*R) @ Phi.T
		return self.H
