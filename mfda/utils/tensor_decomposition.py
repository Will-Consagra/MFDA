from tensorly.decomposition import parafac

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
