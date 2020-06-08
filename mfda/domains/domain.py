from abc import ABC, abstractmethod

import numpy as np 

class Domain(ABC):
	"""
	Abstract object characterizing the (continuous) domains for which functional basis objects are defined over 
	
	Attributes: 

	Notes: 
	"""

	@abstractmethod 
	def __init__(self):
		pass 

	@abstractmethod 
	def _inner_product(self, X1, X2):
		pass 

	@abstractmethod
	def _distance(self, X1, X2):
		pass 

	@abstractmethod
	def in_domain(self, X):	
		"""
		Validate data is in the correct domain.
		"""
		pass 

	def inner_product(self, X1, X2=None):
		"""
		performs required checks 
		"""
		if X2 is None:
			return self._inner_product(X1, X1) 
		else: 
			return self._inner_product(X1, X2)

	def distance(self, X1, X2=None):
		"""
		performs required checks 
		"""
		if X2 is None:
			return self._inner_product(X1, X1) 
		else: 
			return self._inner_product(X1, X2)