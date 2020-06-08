"""Utilities for input validation"""

def _check_tensor_grid_accordance(X, Y):
	"""
	This function checks to make sure the marginal coordinate grid system is in accordance with the dimensions of tensor Y 
	Arguments:
		X: (list of marginal coordinates)
		Y: n_1 x ... x n_d mode array
	Returns: 
		Boolean flag 
	"""
	pass 

def infer_X_data(X):
	"""
	This function determines the nature of the input X data, e.g. whether it is a list of marginal grids or point cloud.
	Raises value error if it does not adhere to a particular structure.
	Arguments:
		X: data whose structure is to be inferred. 
	Returns:
		Indicator of the type of data, reformatted to be a list of np.ndarrays.
	"""
	pass 

def check_is_fitted(basisobject):
	"""
	Checks whether the objects fit function has been called yet.
	Arguments:
		basisobject: inherits from FBasis
	Returns:
		Boolean flag 
	"""
	pass 