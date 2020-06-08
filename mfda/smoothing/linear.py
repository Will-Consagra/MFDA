from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin

class LinearSmoother(ABC, BaseEstimator, TransformerMixin):
	"""
	Abstract class for all linear smoothers 

	"""
	@abstractmethod
	def __init__(self):
		"""
		Placehold for initialization method
		"""
		pass 

	@abstractmethod
	def _hat_matrix(self, X):
		pass 

	def hat_matrix(self, X):
		"""
		X: domain points for which the observations to be smoothed have been calculated. 
		"""
		## perform required checks 
		return self._hat_matrix(X)

	def fit(self, X, y=None):
		"""
		X: domain points for which the observations to be smoothed have been calculated. 
		y: ingored
		"""
		## perform required checks and call hat_matrix, which saves the smoothing matrix
		self.hat_matrix(X)
		return self

	def transform(self, Y, y=None):
		"""
		Y: functional observations corresponding to X which are to be smoothed 
		y: ignored
		"""
		## perform required checks 
		Yhat = self.H @ Y 
		return Yhat 

	def GCV(self, X, Y):
		"""
		Computes the GCV statistic. 
		Arguments:
			X: domain points for which the observations to be smoothed have been calculated. 
			Y: functional observations corresponding to X which are to be smoothed
		"""
		raise NotImplementedError 