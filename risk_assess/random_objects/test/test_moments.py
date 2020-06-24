from ..mvg_moments import *
from ..utils import *
from risk_assess.random_objects.multivariate_normal import MultivariateNormal

def mvg_moment_tensors(mu, sigma, max_order, dimension):
    """

    Args:
        mu ([type]): [description]
        sigma ([type]): [description]
        max_order ([type]): [description]
        dimension ([type]): [description]

    Returns:
        [type]: [description]
    """
    moment_array_funcs, _ = mvg_moment_array_functions(max_order, dimension)
    t = np.zeros(dimension)
    moment_tensors = dict()
    for order, fun in moment_array_funcs.items():
        moment_tensors[order] = np.array(fun(t, mu, sigma)).reshape(order*(2,))
    return moment_tensors

def mvg_moment_arrays(mu, sigma, max_order, dimension):
    moment_array_funcs, _ = mvg_moment_array_functions(max_order, dimension)
    t = np.zeros(dimension)
    moment_arrays = dict()
    for order, fun in moment_array_funcs.items():
        marray = np.array(fun(t, mu, sigma))
        # Flatten the array if its 1d.
        if order == 1:
            marray = marray.flatten()
        moment_arrays[order] = marray
    return moment_arrays

def test_moment_tensor_consistent(tolerance=1e-10):
    """ Test that elements of the moment tensor are consistent. That is,
        for any two moment tensor indices, if they have the same corresponding
        multi-index, then they have the same values.
    """
    # Test parameters.
    max_order = 8
    dimension = 2
    mu = np.array([-2.0, 1.0])
    sigma = np.array([[1.0, 0.1], [0.1, 0.5]])
    MultivariateNormal.compile_moment_functions_up_to(dimension, max_order)
    mvg = MultivariateNormal(mu, sigma)

    moment_tensors = mvg_moment_tensors(mu, sigma, max_order, dimension)
    moment_values = mvg.compute_moments_up_to(max_order)
    for moment_tensor in moment_tensors.values():
        for tensor_idx in np.ndindex(moment_tensor.shape):
            multi_idx = tensor_to_multi_idx(tensor_idx, dimension)
            assert abs(moment_tensor[tensor_idx]-moment_values[multi_idx])<=tolerance

def test_tensor_array_idx_conversion(tolerance=1e-10):
    """ Test the conversions between tensor and array idx work as intended.
        
    Args:
        tolerance ([type], optional): [description]. Defaults to 1e-10.
    """
    # Test parameters.
    max_order = 8
    dimension = 2
    mu = np.array([-2.0, 1.0])
    sigma = np.array([[1.0, 0.1], [0.1, 0.5]])
    moment_arrays = mvg_moment_arrays(mu, sigma, max_order, dimension)
    moment_tensors = mvg_moment_tensors(mu, sigma, max_order, dimension)
    for order in moment_tensors.keys():
        # Get the tensor and array to test.
        moment_tensor = moment_tensors[order]
        moment_array = moment_arrays[order]
        for tensor_idx in np.ndindex(moment_tensor.shape):
            # For each possible tensor idx, get the array idx and check
            # the values are consistent with each other.
            array_idx = tensor_to_array_idx(tensor_idx, dimension)
            assert abs(moment_tensor[tensor_idx] - moment_array[array_idx])<=tolerance
        
        for array_idx in np.ndindex(moment_array.shape):
            # For each possible array idx, get the tensor idx
            # and check the values are consistent with each other.
            tensor_idx = array_to_tensor_idx(array_idx, order, dimension)

            # The array idx may convert to a lower order moment tensor, so get the right one.
            test_moment_tensor = moment_tensors[len(tensor_idx)]
            assert abs(test_moment_tensor[tensor_idx] - moment_array[array_idx])<=tolerance

def test_generate_moment_functions(tolerance = 1e-10):
    """ Test the method generate_mvg_moment_functions, which returns a dictionary
        of functions with keys as multi-indicies and the resulting functions directly
        compute the moments.

    Args:
        tolerance ([type], optional): [description]. Defaults to 1e-10.
    """
    # Test parameters.
    max_order = 8
    dimension = 2
    mu = np.array([-2.0, 1.0])
    sigma = np.array([[1.0, 0.1], [0.1, 0.5]])
    t = np.zeros(dimension)
    MultivariateNormal.compile_moment_functions_up_to(dimension, max_order)
    mvg = MultivariateNormal(mu, sigma)

    # Ground truth data to test.
    ground_truth_moments = mvg.compute_moments_up_to(max_order)

    # Moment functions to test.
    moment_functions = generate_mvg_moment_functions(max_order, dimension)

    for order in range(1, max_order+1):
        possible_multi_idxs = constant_sum_tuples(dimension, order)
        for multi_idx in possible_multi_idxs:
            fun = moment_functions[multi_idx]
            test_moment = fun(t, mu, sigma)
            assert abs(test_moment - ground_truth_moments[multi_idx]) < tolerance

def test_mvg_class():
    # Test parameters.
    max_order = 4
    dimension = 2
    mu = np.array([-2.0, 1.0])
    sigma = np.array([[1.0, 0.1], [0.1, 0.5]])
    # Test a second object to make sure compilation works for all objects.
    mu2 = np.array([-25.0, 1.0])
    sigma2 = np.array([[2.0, 0.1], [0.1, 0.5]])

    # Compile the moment functions.
    MultivariateNormal.compile_moment_functions_up_to(dimension, max_order)
    mvg = MultivariateNormal(mu, sigma)
    mvg2 = MultivariateNormal(mu2, sigma2)
    for order in range(1, max_order + 1):
        possible_multi_idxs = constant_sum_tuples(dimension, order)
        for multi_idx in possible_multi_idxs:
            # Mostly jsut testing that the function executes.
            out = mvg.compute_moment(multi_idx)
            out2 = mvg.compute_moment(multi_idx)
            assert type(out) == float
            assert type(out2) == float