import numpy as np 
import tensorly as tl

from tensorly.decomposition import parafac
from tensorly.decomposition.candecomp_parafac import initialize_factors
from tensorly.tenalg import inner

from functools import reduce 

def CPD(G, K, max_iter=100, tol=1e-08, normalize=False):
	"""
	Vanilla CPD decomposition 

	Parameters: 
		G: (m1 x ... x mp x N) dimensional tensor. 
		K: rank of basis 
		max_iter: maximum number of iterations in ALS algorithm 
		tol: tolerance to exit optimization 
		normalize: boolean to construct the normalized factors 
	Returns: 
		weights: ndarray (K, ) giving the weights for each factor (all 1 if normalize=False)
		CS: list of factors (length p+1), where each element is m_j x K (or N x K for last one) ndarray
	"""
	weights, CS = parafac(G, K, n_iter_max=max_iter, tol=tol, normalize_factors=normalize)
	return weights, CS

def fCP_TPA(G, Jlst, Rlst, SVDs, lambdas, K, max_iter=100, tol=1e-8, init="svd"):
	"""
	Implementation of the functional CP-TPA algorithm from Allen, 2013 (Algorithm 1).
	
	Arguments: 
			G: m_1 x...x m_pxN data tensor in the "tilde space"; i.e. Y X_1 U_1' X_2 U_2'  ... X_P U_P'
			Jlst: length p list of inner product matrices of marginal basis systems 
			Rlst: length p list of roughness penalty matrices of marginal basis systems 
			SVds: length p list of named tuple object holding the SVDs of the basis evaluation matrices 
			lambdas: length p list of marginal roughness penalties 
			K: Rank of basis 
			max_iter: maximum number of iteration in the inner ALS algorithm
			tol: tolerance to exit inner loop 
			init: {"svd", "random"}; see initialize_factors in tensorly 
	Returns: 

	Notes:
		Eventually, we would like to incorporate GCV to select dimension specific penalty parameters. For now, they are considered 
		as fixed model parameters
	"""
	## Get parameters 
	P = len(G.shape[:-1])
	N = G.shape[-1]
	rangeP = list(range(P))
	mlst = [len(SVDs[p].s) for p in rangeP]
	## Initialize factors 
	factors = initialize_factors(G, K, init=init)

	## Get smoother and inverse smoother matrices for regularization 
	SmootherMatInv = [(np.eye(mlst[p])+lambdas[p]*np.diag(1./SVDs[p].s)@SVDs[p].Vt@Rlst[p]@SVDs[p].Vt.T@np.diag(1./SVDs[p].s)) for p in rangeP]
	SmootherMat = [np.linalg.inv(SmootherMatInv[p]) for p in range(P)]

	## Get tilde-transformed inner product matrices: D_d^{-1}V_d^prime J_d V_d D_d^{-1} --> this does not work!! figure out why that is 
	#Jtilde = [np.diag(1./SVDs[p].s)@SVDs[p].Vt@Jlst[p]@SVDs[p].Vt.T@np.diag(1./SVDs[p].s) for p in rangeP]
	Jtilde = [np.eye(mlst[p]) for p in rangeP]
	## Create ndarrays to hold factors 
	Ctilde = [np.zeros((mlst[p], K)) for p in rangeP]
	Smat = np.zeros((G.shape[-1], K))
	scalars = np.zeros(K)
	Ghat = G 

	for k in range(K):
		i = 0 
		delta_residual_norm = np.inf 
		residual_norm_prev = np.inf 
		C_i_norms = [np.sqrt(factors[p][:, k].reshape((1, mlst[p]))@factors[p][:,k].reshape((mlst[p], 1))).item() for p in rangeP]
		C_i = [factors[p][:, k].reshape((mlst[p], 1))/C_i_norms[p] for p in rangeP]
		S_i = factors[-1][:, k].reshape((N, 1)) 
		while (i <= max_iter) and (delta_residual_norm > tol):
			## Create a copy to assess convergence of solutions
			C_i_copy = [C_i[p].copy() for p in rangeP] 
			S_i_copy = S_i.copy()
			## Update smooth factors
			for p in range(P):
				if p == 0:
					mode_less_p = rangeP[1:]
				elif p == P:
					mode_less_p = rangeP[0:-1]
				else:
					mode_less_p = rangeP[0:p] + rangeP[(p+1):]
				C_less_p = [C_i[pp] for pp in mode_less_p]
				C_p_new = SmootherMat[p] @ tl.tenalg.multi_mode_dot(Ghat, C_less_p + [S_i], modes=mode_less_p + [P], transpose=True).reshape((mlst[p], 1))
				normalize_p = np.prod([(C_i[pp].T @ SmootherMatInv[pp] @ C_i[pp]).item() for pp in mode_less_p])
				C_i[p] = C_p_new / normalize_p
			## Update subject factor; right now the accuracy of this form is uncertain.
			S_new = tl.tenalg.multi_mode_dot(Ghat, C_i, modes=rangeP, transpose=True).reshape((N, 1))
			S_i = S_new / np.linalg.norm(S_new)
			## Scale the norms of smooth factors 
			C_i_norms = [np.sqrt(C_i[p].T@C_i[p]).item() for p in rangeP]
			C_i = [C_i[p]/C_i_norms[p] for p in rangeP]
			## Make identifiability constant  
			scalar_K = tl.tenalg.multi_mode_dot(Ghat, C_i + [S_i], list(range(P+1)), transpose=True).item()
			## Check for convergence 
			residual = Ghat - scalar_K*reduce(np.multiply.outer, [C_i[p].ravel() for p in rangeP] + [S_i.ravel()])
			residual_norm = np.sqrt(inner(residual, residual))
			delta_residual_norm = np.abs(residual_norm_prev - residual_norm)
			residual_norm_prev = residual_norm
			i += 1
		print("Inner loop converged after %s iterations with delta residual norm %s" %(i, delta_residual_norm))
		## Remove rank-1 factor 
		Ghat = residual
		## Save factors 
		for p in rangeP:
			Ctilde[p][:, k] = C_i[p].ravel()
		Smat[:, k] = S_i.ravel()
		scalars[k] = scalar_K

	return Ctilde, Smat, scalars





