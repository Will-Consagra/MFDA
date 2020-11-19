from collections import namedtuple 
from itertools import product

import sys 
sys.path.append("..")

import numpy as np
import tensorly as tl

svdtuple = namedtuple("SVD", ["U", "s", "Vt"])

class MPF(object):
	"""
	Marginal product functional model basis system. 

	Attributes: 	    
		K: Rank (number of basis functions)
		marginal_basis_system: list of basis function objects, inheriting from Fbasis
		algorithm: the tensord decomposition algorithm used in fitting the basis "CPD", ... 

	Notes: -- Consider multiple inheritance (FBasis, "Sklearn.predictive_model")
	       -- Different fit methods are hackily thrown together, need to tighten up the control logic at some point 
	"""

	def __init__(self, K, marginal_basis_systems, nmode, algorithm="CPD"):
		self.K = K
		self.marginal_basis_systems = marginal_basis_systems
		self.algo = algorithm
		self.nmode = nmode

	def fit(self, X, Y):
		"""
		Fit the MPF using the appropriate tensor decomposition. 
		Compute and save the required C1, ..., Cp, S matrices.
		Parameters: 
			X: list of ndarrays containing marginal grid coordinates.
			Y: (n1 x n2 x ... x np x N) dimensional tensor 
		Note: All of the transformations should be done within this function.
		"""
		## Need to implement a check that the data is of the correct dimensions and contained 
		## within the marginal product domain defined by the marginal domains in marginal_basis_systems
		## Check the tensor is the correct size and in accordance with the domain 
		N = Y.shape[-1]
		## Construct the SVD of all the basis matrices 
		nmode = self.nmode
		Phis = [self.marginal_basis_systems[d].evaluate(X[d]) for d in range(nmode)]
		Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]
		## Perform nmode-mode multiplication on Y to obtain transformed tensor G, e.g. G = Y X_1 U_1' X_2 U_2' ... X_nmode U_nmode'
		G = tl.tenalg.multi_mode_dot(Y, [svdt.U.T for svdt in Svds], list(range(nmode)))
		if N == 1: ## Single subject case
			G = G.reshape(G.shape[:-1])
		if len(G.shape) == 2:## Matrix case, just perform the SVD 
			Gu, Gs, Gvt = np.linalg.svd(G) 
			Gv = Gvt.T
			self.Clist = [Gu[:, 0:self.K], Gv[:, 0:self.K]]
			self.Smat = Gs[0:self.K].reshape((1, self.K))
		else: ## Tensor case, use specified tensor decomposition algorithm 
			raise NotImplementedError
			"""
			if self.algo == "CPD":
				weights, Cs = CPD(G, self.K, max_iter=100, tol=1e-08, normalize=False) ## want to allow passing in of **self.algo_params
				self.Clist = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Cs[d] for d in range(nmode)] 
				if N == 1:
					self.Smat = np.ones((1, self.K))
				else:
					self.Smat = Cs[-1]
			else:
				raise NotImplementedError
			"""
		return self

	def fit_symmetric_2D(self, X, Y):
		"""
		Temporary method for exploratory purposes, will eventually be accomaded in classes main functionality.
		"""
		assert self.nmode == 2, "nmode > 2 not yet implemented!"
		nmode = self.nmode
		Phis = [self.marginal_basis_systems[0].evaluate(X[d]) for d in range(nmode)]
		Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]
		G = Svds[0].U.T @ Y @ Svds[1].U 
		S, C = np.linalg.eig(G)
		idx = S.argsort()[::-1]
		S = np.real(S[idx])
		C = np.real(C[:, idx])
		self.Clist  = [Svds[0].Vt.T @ np.diag(1/Svds[0].s) @ C[:, 0:self.K], 
						Svds[1].Vt.T @ np.diag(1/Svds[1].s) @ C[:, 0:self.K]]
		self.Smat = S[0:self.K]
		return self 

	def _Xi(self, X):
		"""
		X: list of marginal grids or list or point cloud
		Note: I Think we are mistting smat!!!!!
		"""
		nmode = self.nmode
		if isinstance(X, list): ## list of marginal grid points 
			Xi = [np.prod(np.ix_(*[self.marginal_basis_systems[d].evaluate(X[d]) @ self.Clist[d][:, k] for d in range(nmode)])) for k in range(self.K)]
		elif isinstance(X, np.ndarray):
			Xi = [np.prod(np.array([self.marginal_basis_systems[d].evaluate(X[:,d]) @ self.Clist[d][:, k] for d in range(nmode)]),axis=1) for k in range(self.K)]
		else:
			raise ValueError
		return Xi

	def predict(self, X):
		"""
		X: list of marginal grids or list or point cloud
		"""
		## Implement correct checks later 
		Xi = self._Xi(X)
		return sum(Xi)

	def _evaluate(self, X):
		raise NotImplementedError

	def _gradient(self):
		raise NotImplementedError 

	def _inner_product_matrix(self, other=None):
		raise NotImplementedError 

	def _roughness_matrix(self):
		"""
		Pairwise inner product of Laplacian
		"""
		raise NotImplementedError 

	@property
	def training_ceofs(self):
		## Implement check to see if fit has been called yet 
		return self.Smat
	