
import numpy as np
import tensorly as tl 
from sklearn.decomposition import PCA

def _mpf_inner_product_matrix(marginal_basis_systems, Clist, scalars, J_phi):
	"""
	Note: This function will eventually go into mpf class 
	"""
	K = len(scalars)
	nmode = len(Clist)
	#J_phi = [marginal_basis_systems[d].gram_matrix() for d in range(nmode)]
	mlst = [J_phi[d].shape[0] for d in range(nmode)]
	J_psi = np.zeros((K, K))
	for k1 in range(K):
		for k2 in range(k1, K):
			J_psi[k1, k2] = scalars[k1]*scalars[k2]*np.prod(
					[(Clist[d][:, k1].reshape((1, mlst[d]))@J_phi[d]@Clist[d][:, k2].reshape((mlst[d], 1))).item() for d in range(nmode)]
															)
	ix_low_tri = np.tril_indices(K, -1)
	J_psi[ix_low_tri] = J_psi.T[ix_low_tri] 
	assert np.allclose(J_psi, J_psi.T), "Inner product matrix must be symmetric!!" 
	return J_psi

def _mpf_roughness_matrix(marginal_basis_systems, Clist, scalars, J_phi, J_D2phi, J_phiD2phi):
	"""
	Note: This function will eventually fo into the mpf class. Also, check derivation on this before "going public" 
	"""
	K = len(scalars)
	nmode = len(Clist)
	#J_phi = [marginal_basis_systems[d].gram_matrix() for d in range(nmode)]
	#J_D2phi = [marginal_basis_systems[d].roughness_matrix() for d in range(nmode)]
	#J_phiD2phi = [marginal_basis_systems[d].cross_roughness_matrix() for d in range(nmode)]
	mlst = [J_phi[d].shape[0] for d in range(nmode)]
	R_psi = np.zeros((K, K))
	for i in range(K):
		for j in range(i, K):
			for d in range(nmode):
				for dprime in range(nmode):
					if d == dprime:
						Prod_less_d =  np.prod([Clist[dtilde][:,j].reshape((1,mlst[dtilde]))@J_phi[dtilde]@Clist[dtilde][:,i].reshape((mlst[dtilde],1)) \
								                            for dtilde in range(nmode) if dtilde not in (d,)])
						R_psi[i,j] = R_psi[i,j] + Prod_less_d * \
										Clist[d][:,j].reshape((1,mlst[d]))@J_D2phi[d]@Clist[d][:,i].reshape((mlst[d],1))
					else:
						Prod_d_dprime = Clist[d][:,j].reshape((1,mlst[d]))@J_phiD2phi[d]@Clist[d][:,i].reshape((mlst[d],1)) * \
							     Clist[dprime][:,i].reshape((1,mlst[dprime]))@J_phiD2phi[dprime]@Clist[dprime][:,j].reshape((mlst[dprime],1))
						if nmode > 2:
							Prod_less_d_dprime = np.prod([Clist[dtilde][:,j].reshape((1,mlst[dtilde]))@J_phi[dtilde]@Clist[dtilde][:,i].reshape((mlst[dtilde],1)) \
								                            for dtilde in range(nmode) if dtilde not in (d, dprime)])
							R_psi[i,j] = R_psi[i,j] + Prod_less_d_dprime * Prod_d_dprime
						else:
							R_psi[i,j] =  R_psi[i,j] + Prod_d_dprime
			R_psi[i,j] = scalars[i]*scalars[j]*R_psi[i,j]
	ix_low_tri = np.tril_indices(K, -1)
	R_psi[ix_low_tri] = R_psi.T[ix_low_tri] 
	assert np.allclose(R_psi, R_psi.T), "Roughenss matrix must be symmetric!!" 
	return R_psi

def post_hoc_fpca(S, J_psi, R_psi, lam=0.0):
	"""
	Arguments:
		S: (ndarray) N x K matrix of subject coefficients w.r.t. the marginal product basis functions
		J_psi: (ndarray) KxK L2 inner product matrix of marginal product basis functions
		R_psi: (ndarray) KxK L2 inner product matrix of linear differential operator penalty functional applied to marginal product basis functions 
		lam: (float) regularization strength, if lam < 0 -> Just perform basis orthogonalization
	Returns:

	Notes:
	"""
	S_centered = S - np.mean(S, 0)
	K = J_psi.shape[0]
	if lam < 0.0:
		N = S_centered.shape[0]
		Sigma_S = (1./N) * S_centered.T @ S_centered
		lambdas, V = np.linalg.eig(J_psi)
		J_psi_sqrt = V @ np.diag(np.sqrt(lambdas)) @ V.T
		J_psi_sqrt_inv = V @ np.diag(1./np.sqrt(lambdas)) @ V.T
		gamma, Btilde = np.linalg.eig(J_psi_sqrt @ Sigma_S @ J_psi_sqrt)
		B = J_psi_sqrt_inv @ Btilde
	else:
		M_lambda = J_psi + lam*R_psi
		L = np.linalg.cholesky(M_lambda)
		SL = np.linalg.inv(L.T)
		D = SL.T @ J_psi @ S_centered.T
		D_pca = PCA()
		D_pca.fit(D.T)
		B = SL @ D_pca.components_.T
		for k in range(B.shape[1]):
			#B[:, k] = B[:, k] / np.sqrt(B[:, k].reshape((1,K))@J_psi@B[:, k].reshape((K,1))).item()
			B[:, k] = B[:, k] / np.sqrt(B[:, k].reshape((1,K))@B[:, k].reshape((K,1))).item()
		gamma = D_pca.singular_values_
	return B, gamma

def ridge_smoother(Psi_tensor, Y_new, R_psi, lam):
	"""
	Performs ridge smoothing, w.r.t. differential operator determining R_psi, for basis system psi 
	Arguments:
		Psi_tensor: K X n_1 X ... X n_D tensor of the K mpf basis functions evaluated over observation grid 
		Y_new: n_1 X ... X n_D tensor of observations, considered to be a new realization from distribution psi was trained on
		R_psi: KxK matrix of pairwise inner product of linear differential operator (usually laplacian)
		lam: float defining the regularization strength.
	Returns: 
		H_K: (n1*...*nd) X (n1*...*nd) ridge smoother matrix 
		coefs: Kx1 vector of coefficients, i.e. coordinates of representation of Y_new w.r.t. mpf basis system psi 
	Notes: 
		Can only be used with tensor so large, a good place to look for some slick numerical approximation procedures.
	"""
	y = tl.base.tensor_to_vec(Y_new)
	Nobs = len(y)
	Psi_K = tl.base.unfold(Psi_tensor, 0).T
	store_inv = np.linalg.inv(Psi_K.T @ Psi_K + lam*R_psi)
	H_K = Psi_K @ store_inv @ Psi_K.T 
	coefs = store_inv @ Psi_K.T @ y 
	## compute gcv 
	I = np.identity(Nobs)
	SSE = np.sum(np.power((I - H_K) @ y, 2))
	residual_space_trace = np.sum(np.diag(I - H_K))
	gcv = (SSE/Nobs)/np.power(residual_space_trace/Nobs, 2)
	return H_K, coefs, gcv

def PVE(J_psi, Sigma_S, B):
	"""
	Arguments:
		J_psi: (ndarray) KxK L2 inner product matrix of marginal product basis functions
		Sigma_S: (ndarry) KxK covariance matrix of the S's 
		B: (ndarry) K:K matrix of (columnwise) coefficients for etas 
	returns:
		pve: (array) lenght K list where each element is the estimated proportion of variance explained by the first K etas 
	"""
	K = B.shape[0]
	ve = np.zeros(K)
	for k in range(1, K):
		Lambda_K = J_psi@ B[:, 0:k] @ np.linalg.inv(B[:, 0:k].T @ J_psi @ B[:, 0:k]) @ B[:, 0:k].T @ J_psi
		ve[k] = np.trace(Lambda_K @ Sigma_S @ Lambda_K @ Sigma_S)
	pve = ve / np.trace(J_psi @ Sigma_S @ J_psi @ Sigma_S)
	return pve 
