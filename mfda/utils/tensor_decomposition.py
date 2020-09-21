import datetime 
import numpy as np 
import tensorly as tl

from tensorly.decomposition import parafac
from tensorly.decomposition.candecomp_parafac import initialize_factors
from tensorly.tenalg import inner

from functools import reduce 

def CPD_ALS(G, K, max_iter=100, tol=1e-08, normalize=False):
	"""
	CPD decomposition via ALS algorithm

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
	scalars, CS = parafac(G, K, n_iter_max=max_iter, tol=tol, normalize_factors=normalize)
	Ctilde = [CS[d] for d in range(len(G.shape[:-1]))]
	Smat = CS[-1]
	return Ctilde, Smat, scalars

def fCP_TPA_allen(Y, PenMat, lambdas, K, max_iter=100, tol=1e-8, init="svd"):
	"""
	Implementation of functional CP-TPA algorithm from Allen, 2013 (Algorithm 1).

	Arguments:
			Y: n_1 x ... x n_p X N raw data tensor 
			PenMat: lenght P list of n_dxn_d matrices defining the smoothing structure to be appleid to factors
			lambdas: length p list of marginal roughness penalties
			K: Rank of decomposition 
			max_iter: maximum number of iteration in the inner ALS algorithm
			tol: tolerance to exit inner loop 
			init: {"svd", "random"}; see initialize_factors in tensorly 

	Returns: 
	"""
	P = len(Y.shape[:-1])
	N = Y.shape[-1]
	rangeP = list(range(P))
	nlst = Y.shape[:-1]
	## Initialize factors 
	factors = initialize_factors(Y, K, init=init)
	## Get inverse smoothermat for regularization 
	SmootherMatInv = [(np.eye(nlst[p])+lambdas[p]*PenMat[p]) for p in rangeP]
	SmootherMat = [np.linalg.inv(SmootherMatInv[p]) for p in range(P)]
	## Create ndarrays to hold factors 
	Clst = [np.zeros((nlst[p], K)) for p in rangeP]
	Smat = np.zeros((Y.shape[-1], K))
	scalars = np.zeros(K)
	Yhat = Y
	## Perform decomposition
	for k in range(K):
		i = 0 
		delta_residual_norm = np.inf 
		residual_norm_prev = np.inf 
		C_i_norms = [np.sqrt(factors[p][:, k].reshape((1, nlst[p]))@factors[p][:,k].reshape((nlst[p], 1))).item() for p in rangeP]
		C_i = [factors[p][:, k].reshape((nlst[p], 1))/C_i_norms[p] for p in rangeP]
		S_i = factors[-1][:, k].reshape((N, 1)) 
		while (i <= max_iter) and (delta_residual_norm > tol):
			## Update smooth factors
			for p in range(P):
				if p == 0:
					mode_less_p = rangeP[1:]
				elif p == P:
					mode_less_p = rangeP[0:-1]
				else:
					mode_less_p = rangeP[0:p] + rangeP[(p+1):]
				C_less_p = [C_i[pp] for pp in mode_less_p]
				C_p_new = SmootherMat[p] @ tl.tenalg.multi_mode_dot(Yhat, C_less_p + [S_i], modes=mode_less_p + [P], transpose=True).reshape((nlst[p], 1))
				normalize_p = np.prod([(C_i[pp].T @ SmootherMatInv[pp] @ C_i[pp]).item() for pp in mode_less_p])
				C_i[p] = C_p_new / normalize_p
			## Update subject factor; right now the accuracy of this form is uncertain.
			S_new = tl.tenalg.multi_mode_dot(Yhat, C_i, modes=rangeP, transpose=True).reshape((N, 1))
			S_i = S_new / np.linalg.norm(S_new)
			## Scale the norms of smooth factors 
			C_i_norms = [np.sqrt(C_i[p].T@C_i[p]).item() for p in rangeP]
			C_i = [C_i[p]/C_i_norms[p] for p in rangeP]
			## Make identifiability constant  
			scalar_K = tl.tenalg.multi_mode_dot(Yhat, C_i + [S_i], list(range(P+1)), transpose=True).item()
			## Check for convergence 
			residual = Yhat - scalar_K*reduce(np.multiply.outer, [C_i[p].ravel() for p in rangeP] + [S_i.ravel()])
			residual_norm = np.sqrt(inner(residual, residual))
			delta_residual_norm = np.abs(residual_norm_prev - residual_norm)
			residual_norm_prev = residual_norm
			i += 1
		print("Inner loop converged after %s iterations with delta residual norm %s" %(i, delta_residual_norm))
		## Remove rank-1 factor 
		Yhat = residual
		## Save factors 
		for p in rangeP:
			Clst[p][:, k] = C_i[p].ravel()
		Smat[:, k] = S_i.ravel()
		scalars[k] = scalar_K

	return Clst, Smat, scalars

def fCP_TPA_defunct(G, Jlst, Rlst, SVDs, lambdas, K, max_iter=100, tol=1e-8, init="svd"):
	"""
	Augmentation of the functional CP-TPA algorithm from Allen, 2013 (Algorithm 1) to work with MPF basis. 
	This implementation should not be used and should eventually be deleted.
	
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
		
	"""
	## Get parameters 
	P = len(G.shape[:-1])
	N = G.shape[-1]
	rangeP = list(range(P))
	mlst = [len(SVDs[p].s) for p in rangeP]
	## Initialize factors 
	factors = initialize_factors(G, K, init=init)

	## Get tilde-transformed inner product matrices: D_d^{-1}V_d^prime J_d V_d D_d^{-1} --> this does not work!! figure out why that is 
	Jtilde = [np.diag(1./SVDs[p].s)@SVDs[p].Vt@Jlst[p]@SVDs[p].Vt.T@np.diag(1./SVDs[p].s) for p in rangeP]

	## Get smoother and inverse smoother matrices for regularization 
	SmootherMatInv = [(np.eye(mlst[p])+lambdas[p]*np.diag(1./SVDs[p].s)@SVDs[p].Vt@Rlst[p]@SVDs[p].Vt.T@np.diag(1./SVDs[p].s)) for p in rangeP]
	SmootherMat = [np.linalg.inv(SmootherMatInv[p]) for p in range(P)]
	## Create ndarrays to hold factors 
	Ctilde = [np.zeros((mlst[p], K)) for p in rangeP]
	Smat = np.zeros((G.shape[-1], K))
	scalars = np.zeros(K)
	Ghat = G 
	
	## Perform decomposition
	for k in range(K):
		i = 0 
		delta_residual_norm = np.inf 
		residual_norm_prev = np.inf 
		C_i_norms = [np.sqrt(factors[p][:, k].reshape((1, mlst[p]))@factors[p][:,k].reshape((mlst[p], 1))).item() for p in rangeP]
		C_i = [factors[p][:, k].reshape((mlst[p], 1))/C_i_norms[p] for p in rangeP]
		S_i = factors[-1][:, k].reshape((N, 1)) 
		while (i <= max_iter) and (delta_residual_norm > tol):
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


def fCP_TPA_gcv(G, Jlst, Rlst, SVDs, K, lambda_grid, lambda_init="random", max_iter=100, tol=1e-8, init="svd"):
	"""
	Implementation of the functional CP-TPA algorithm from Allen, 2013 (Algorithm 1) with higher mode adaptation of 
	GCV selection procedure given in Huang, 2009.
	
	Arguments: 
			G: m_1 x...x m_pxN data tensor in the "tilde space"; i.e. Y X_1 U_1' X_2 U_2'  ... X_P U_P'
			Jlst: length p list of inner product matrices of marginal basis systems 
			Rlst: length p list of roughness penalty matrices of marginal basis systems 
			SVds: length p list of named tuple object holding the SVDs of the basis evaluation matrices 
			lambda_grids: length p list of grids of roughness penalties to select from for each mode
			lambda_init: how to initialize lambdas, for now only random initialization is accepted 
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
	lambda_inits = np.zeros((P, K))
	for p in rangeP:
		lambda_inits[p, :] = np.random.choice(lambda_grid[p], size=K)
	## Get smoother and inverse smoother matrices for regularization 
	
	SmootherMat = {}
	SmootherMatInv = {}
	for p in rangeP:
		SmootherMatInv[p] = {lam:(np.eye(mlst[p])+lam*np.diag(1./SVDs[p].s)@SVDs[p].Vt@Rlst[p]@SVDs[p].Vt.T@np.diag(1./SVDs[p].s)) for lam in lambda_grid[p]}
		SmootherMat[p] = {lam:np.linalg.inv(SmootherMatInv[p][lam]) for lam in lambda_grid[p]}
	Imat = [np.eye(mlst[p]) for p in rangeP]

	## Create ndarrays to hold factors 
	Ctilde = [np.zeros((mlst[p], K)) for p in rangeP]
	Smat = np.zeros((G.shape[-1], K))
	scalars = np.zeros(K)
	lambdas = np.zeros((len(lambda_grid), K))
	Ghat = G 

	for k in range(K):
		i = 0 
		delta_residual_norm = np.inf 
		residual_norm_prev = np.inf 
		C_i_norms = [np.linalg.norm(factors[p][:, k]) for p in rangeP]
		C_i = [factors[p][:, k].reshape((mlst[p], 1))/C_i_norms[p] for p in rangeP]
		S_i = factors[-1][:, k].reshape((N, 1)) 
		lambda_i = lambda_inits[:, k]
		## Get smoother and inverse smoother matrices for regularization 
		H_i_Inv = [SmootherMatInv[p][lambda_i[p]] for p in rangeP]
		while (i <= max_iter) and (delta_residual_norm > tol):
			## Update smooth factors
			for p in range(P):
				if p == 0:
					mode_less_p = rangeP[1:]
				elif p == P:
					mode_less_p = rangeP[0:-1]
				else:
					mode_less_p = rangeP[0:p] + rangeP[(p+1):]
				C_less_p = [C_i[pp] for pp in mode_less_p]
				C_p_wo = tl.tenalg.multi_mode_dot(Ghat, C_less_p + [S_i], modes=mode_less_p + [P], transpose=True).reshape((mlst[p], 1))
				C_p_wo_norm = np.linalg.norm(C_p_wo)
				C_p_wo_normed = C_p_wo/C_p_wo_norm
				normalize_p = np.prod([(C_i[pp].T @ H_i_Inv[pp] @ C_i[pp]).item() for pp in mode_less_p])
	
				gcv_p = np.zeros(len(lambda_grid[p])) 
				for ilam, lam in enumerate(lambda_grid[p]):
					H_p_lambda = SmootherMatInv[p][lam]
					Pseudo_hat_p = H_p_lambda/( normalize_p/C_p_wo_norm )
					gcv_num = (1./mlst[p])*np.linalg.norm((Imat[p] - Pseudo_hat_p)@C_p_wo_normed)**2
					gcv_denom = (1. - (1./mlst[0])*np.trace(Pseudo_hat_p))**2 
					gcv_p[ilam] = gcv_num/gcv_denom
				lambda_i[p] = lambda_grid[p][np.argmin(gcv_p)]
				#lambda_i[p] = 1e-5
				H_p_lambda_optim = SmootherMat[p][lambda_i[p]]
				C_p_new = H_p_lambda_optim @ C_p_wo
				C_i[p] = C_p_new / normalize_p
			## Update subject factor; right now the accuracy of this form is uncertain.
			S_new = tl.tenalg.multi_mode_dot(Ghat, C_i, modes=rangeP, transpose=True).reshape((N, 1))
			S_i = S_new / np.linalg.norm(S_new)
			## Scale the norms of smooth factors 
			C_i_norms = [np.linalg.norm(C_i[pp]) for pp in rangeP]
			C_i = [C_i[pp]/C_i_norms[pp] for pp in rangeP]
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
		lambdas[:, k] = lambda_i

	return Ctilde, Smat, scalars, lambdas

def fCP_TPA(G, Jlst, Rlst, SVDs, lambdas, K, max_iter=100, tol=1e-8, init="svd", verbose=False):
	"""
	Augmentation of the functional CP-TPA algorithm from Allen, 2013 (Algorithm 1) to work with MPF basis.
	
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
			verbose: Boolean specifying whether to print run-time updates 
	Returns: 
			Ctilde: List of numpy arrays of ctilde factors 
			Smat: numpy array of subject coefficients 
			scalars: numpy array of scaling factors 
	Notes:
		
	"""
	## Get parameters 
	P = len(G.shape[:-1])
	N = G.shape[-1]
	Ps = list(range(P))
	mlst = [len(SVDs[p].s) for p in Ps]
	## Initialize factors 
	factors = initialize_factors(G, K, init=init)
	## Get smoother and inverse smoother matrices for regularization 
	SmootherMatInv = [(np.eye(mlst[p])+lambdas[p]*np.diag(1./SVDs[p].s)@SVDs[p].Vt@Rlst[p]@SVDs[p].Vt.T@np.diag(1./SVDs[p].s)) 
														for p in Ps]
	SmootherMat = [np.linalg.inv(SmootherMatInv[p]) for p in Ps]
	## Create ndarrays to hold factors 
	Ctilde = [np.zeros((mlst[p], K)) for p in Ps]
	Smat = np.zeros((G.shape[-1], K))
	scalars = np.zeros(K)
	Ghat = G 
	## Perform rank K decomposition
	for k in range(K):
		i = 0 
		delta_residual_norm = np.inf 
		residual_norm_prev = np.inf 
		C_i_norms = [np.linalg.norm(factors[p][:, k]) for p in Ps]
		C_i = [factors[p][:, k].reshape(mlst[p], 1)/C_i_norms[p] for p in Ps]
		S_i = factors[-1][:, k].reshape(N, 1) 
		## Iteratively update block coordinate solutions until convergence.
		while (i <= max_iter) and (delta_residual_norm > tol):
			## Update smooth factors
			for p in range(P):
				if p == 0:
					mode_less_p = Ps[1:]
				elif p == P:
					mode_less_p = Ps[0:-1]
				else:
					mode_less_p = Ps[0:p] + Ps[(p+1):]
				C_less_p = [C_i[pp] for pp in mode_less_p]
				C_p_new = SmootherMat[p] @ tl.tenalg.multi_mode_dot(Ghat, C_less_p + [S_i], 
																	modes=mode_less_p + [P], transpose=True).reshape(mlst[p], 1)
				normalize_p = np.prod([(C_i[pp].T @ SmootherMatInv[pp] @ C_i[pp]).item() for pp in mode_less_p])
				C_i[p] = C_p_new / normalize_p
			## Update subject factor
			S_new = tl.tenalg.multi_mode_dot(Ghat, C_i, modes=Ps, transpose=True).reshape(N, 1)
			S_i = S_new / np.linalg.norm(S_new)
			## Scale the norms of smooth factors 
			C_i_norms = [np.linalg.norm(C_i[p]) for p in Ps]
			C_i = [C_i[p]/C_i_norms[p] for p in Ps]
			## Make identifiability constant  
			scalar_K = tl.tenalg.multi_mode_dot(Ghat, C_i + [S_i], list(range(P+1)), transpose=True).item()
			## Check for convergence 
			residual = Ghat - scalar_K*reduce(np.multiply.outer, [C_i[p].ravel() for p in Ps] + [S_i.ravel()])
			residual_norm = np.sqrt(inner(residual, residual))
			delta_residual_norm = np.abs(residual_norm_prev - residual_norm)
			residual_norm_prev = residual_norm
			i += 1
		if verbose:
			print("%s: Inner loop terminated after %s iterations with delta residual norm %s" %(datetime.datetime.now(),
																								i, 
																								delta_residual_norm))
		## Remove rank-1 factor 
		Ghat = residual
		## Save factors 
		for p in Ps:
			Ctilde[p][:, k] = C_i[p].ravel()
		Smat[:, k] = S_i.ravel()
		scalars[k] = scalar_K
	return Ctilde, Smat, scalars

