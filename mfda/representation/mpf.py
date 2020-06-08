from skfda.representation.basis import BSpline, Fourier
from collections import namedtuple 
from itertools import product

from .basis import FBasis
import sys 
sys.path.append("..")
from utils.tensor_decomposition import CPD

import numpy as np
import tensorly as tl

MBS = {"BSpline":BSpline, 
		"Fourier":Fourier}
svdtuple = namedtuple("SVD", ["U", "s", "Vt"])

class MPF(FBasis):
	"""
	Marginal product functional model basis system. 

	Attributes: 	    
		domain: Domain object 
		K: Rank (number of basis functions)
		mb_dims = array like ==> number of basis functions in each marginal basis system
		mb_type = string (type of marginal basis system) ["BSpline", "Fourier"]
		algorithm = the tensord decomposition algorithm used in fitting the basis "CPD", ... 

	Notes: -We take inspiration from the sklearn fit - predict api. An object of class MPF can be instantiated but must be 
		   "fit" before using methods such as evaluate and so on. Must implement a similiar method to check_is_fitted in sklearn.
		   -All of the fit parameters should be passed through the __init__ function.
		   -Should be able to pass in parameters to marginal basis and to the tensor decomposition algorithm
	"""

	def __init__(self, domain, K, mb_dims, mb_type="BSpline", algorithm="CPD"):
		super().__init__(domain) 
		## perform checks between marginal_basis and the domain to make sure they are in accordance.
		self.K = K
		marginal_domains = domain.marginal_domains()
		self.marginal_basis_systems = [MBS[mb_type](domain_range=marginal_domains[i], n_basis=mb_dims[i]) for i in range(domain.p)]
		self.algo = algorithm

	def fit(self, X, Y):
		"""
		Fit the MPF using the appropriate tensor decomposition. 
		Compute and save the required C1, ..., Cp, S matrices.
		Parameters: 
			X: list of ndarrays containing marginal grid coordinates.
			Y: (n1 x n2 x ... x np x N) dimensional tensor 
		Note: All of the transformations should be done within this function.
		"""
		if not self.domain.in_domain(X):
			raise ValueError 
		## Check the tensor is the correct size and in accordance with the domain 
		N = Y.shape[-1]
		## Construct the SVD of all the basis matrices 
		nmode = self.domain.dim
		Phis = [self.marginal_basis_systems[d].evaluate(X[d]).T for d in range(nmode)]
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
			if self.algo == "CPD":
				weights, Cs = CPD(G, self.K, max_iter=100, tol=1e-08, normalize=False) ## want to allow passing in of **self.algo_params
				self.Clist = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Cs[d] for d in range(nmode)] 
				if N == 1:
					self.Smat = np.ones((1, self.K))
				else:
					self.Smat = Cs[-1]
			else:
				raise NotImplementedError
		return self

	def _Xi(self, X):
		"""
		X: list of marginal grids or list or point cloud
		"""
		nmode = self.domain.dim
		if isinstance(X, list): ## list of marginal grid points 
			Xi = [np.prod(np.ix_(*[self.marginal_basis_systems[d].evaluate(X[d]).T @ self.Clist[d][:, k] for d in range(nmode)])) for k in range(self.K)]
		elif isinstance(X, np.ndarray):
			Xi = [np.prod(np.array([self.marginal_basis_systems[d].evaluate(X[:,d]).T @ self.Clist[d][:, k] for d in range(nmode)]),axis=1) for k in range(self.K)]
		else:
			raise ValueError
		return Xi

	def _evaluate(self, X):
		"""
		X: list of marginal grids or list or point cloud
		"""
		## Implement correct checks later 
		Xi = self._Xi(X)
		return sum(Xi)

	def _gradient(self):
		pass 

	def _inner_product_matrix(self, other=None):
		pass 

	def _roughness_matrix(self):
		"""
		Pairwise inner product of Laplacian
		"""
		pass 

	@property
	def training_ceofs(self):
		## Implement check to see if fit has been called yet 
		return self.Smat
	