import math
import sympy as sp
import string

def constant_sum_tuples(length, total_sum):
    """ Generate all tuples that sum to a given value.

    Args:
        length ([type]): [description]
        total_sum ([type]): [description]

    Yields:
        [type]: [description]
    """
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in constant_sum_tuples(length - 1, total_sum - value):
                yield (value,) + permutation

def int2base(x, base):
    """ Convert a base 10 integer into a new base.

    Args:
        x (int): integer to convert.
        base (int): base to convert to.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        tuple of ints: 
    """
    if x == 0:
        return (0,)
    elif x < 0:
        raise Exception("Input must be non-negative.")
    elif base > 9:
        raise Exception("int2base currently does not support base > 9.")

    digits = []

    while x:
        digits.append(int(x % base))
        x = int(x / base)

    digits.reverse()
    return tuple(digits)

def tensor_to_multi_idx(tensor_idx, dimension):
    return tuple(tensor_idx.count(i) for i in range(dimension))

def multi_to_tensor_idx(multi_idx, dimension):
    # Build the index of the corresponding element in the moment tensor.
    tensor_idx = tuple()
    for var_number in range(dimension):
        # var_number corresponds to an element of the random vector.
        # The order of the moment of "var" corresponds to the number of times 
        # the index of "var" shows up in the index of the moment matrix.
        tensor_idx += multi_idx[var_number]*(var_number,)
    return tensor_idx

def array_to_tensor_idx(array_idx, moment_order, dimension):
    """ Given a moment array index, return the corresponding tensor
        index. This essentially comes down to transforming the first
        element of array_idx into a base-dimension number reprsented
        as a tuple and then inserting the second element of array_idx into the
        tuple. To see this is how it should be done, consider the relationship
        between a 2x2x2 moment tensor and its corresponding 4x2 moment array.

    Args:
        array_idx (2-tuple of ints): index in the moment array.
        moment_order (int): order of the moments array_idx corresponds to.
        dimension (int): dimension of the random vector.
    Returns:
        tuple of ints: index in the moment tensor
    """
    tensor_idx = int2base(array_idx[0], dimension)

    # If 1D array, we just return the tensor idx.
    if len(array_idx)==1:
        return tensor_idx

    # Otherwise, we need to insert the second dimension idx.
    tensor_idx = list(tensor_idx)
    tensor_idx.insert(1, array_idx[1])

    # Fill with zeros so the dimension is consistent.
    if moment_order > len(tensor_idx):
        extra_zeros = (moment_order - len(tensor_idx)) * [0]
        tensor_idx += extra_zeros

    return tuple(tensor_idx)

def tensor_to_array_idx(tensor_idx, dimension):
    """ Given a moment tensor index, return the corresponding index in the
        2D array representation of the tensor. This essentially comes down
        to vieweing tensor_idx as a base-"dimension" number that should be
        converted to decimal form, with a slight twist: we always set the
        second dimension of tensor_idx to 0 in the conversion. This is because
        the array form has two dimensions, it's a bit strange.

    Args:
        tensor_idx (tuple of ints): tensor index.
        dimension (positive int): dimension of the random vector.
    """
    assert len(tensor_idx) > 0

    # The second dimension of the tensor does not get included
    # in the conversion. To see this, consider the 2x2x2 case, where
    # the array representation is 4x2.     decimal_expansion = 0
    reduced_tensor_idx = list(tensor_idx)
    if len(reduced_tensor_idx) > 1:
        del reduced_tensor_idx[1]

    # Make the conversion based off # https://www.rapidtables.com/convert/number/binary-to-decimal.html.
    decimal_expansion = 0
    for i, idx in enumerate(reduced_tensor_idx):
        decimal_expansion += idx * dimension**i
    
    assert dimension > 0
    if dimension == 1 or len(tensor_idx)==1:
        return (decimal_expansion,)
    else:
        return (decimal_expansion, tensor_idx[1])
        
def chi_square_moments(order, dof):
    # https://en.wikipedia.org/wiki/Chi-square_distribution#Noncentral_moments
    num = 2**order * math.gamma(order + 0.5*dof)
    denom = math.gamma(0.5*dof)
    return num/denom

def offset_moments(moments, offset):
    max_order = max(moments.keys())
    m = sp.Symbol("m")
    offset_moments = dict()
    for order in range(1, max_order+1):
        offset_expression = sp.poly((m + offset)**order, m)
        # Substitute moments accordingly.
        for i in range(order, 0, -1):
            offset_expression = offset_expression.subs(m**i, moments[i])
        offset_moments[order] = float(offset_expression)
    return offset_moments