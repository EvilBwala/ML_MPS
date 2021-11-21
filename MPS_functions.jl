using Statistics
using ITensors

"""
Input sequences will be denoted by X
Output sequences will be denoted by Y
The MPO transforming x to y will be denoted by W
"""

"""
This function optimizes W at index idx
"""
function single_site_optimizer(X::Vector{MPS}, Y::Vector{MPS}, W::MPO, idx::Int64, alpha::Float64)::ITensor
    # X, Y are the input data and output data respectively
    # W is the MPO we need to optimize, idx is the index to be optimized, alpha is the regularizer coeficient
    # First check if elements x and w, w and y have matching indices
    # Next form the tensors, A,B,C,D,F,L,R,U
    # Now reshape the tensors F + alpha*C*D and U into matrix and vector respectively
    # Next find the various elements of W_idx by inverting F + alpha*C*D matrix and multiplying it with U
    # Reshape back the vector W into an appropriate tensor
    # Return W
end


"""
This function runs one sweep
"""
function one_sweep(X::Vector{MPS}, Y::Vector{MPS}, W::MPO, alpha::Float64)::MPO
    L = length(X)
    # Forward sweep
    for i in 1:L
        W[i] = single_site_optimizer(X, Y, W, i, alpha)
    end
    # Backward sweep
    for i in i:L
        W[L-i+1] = single_site_optimizer(X, Y, W, L-i+1, alpha)
    end
    return W
end

