using Statistics
using ITensors
using LinearAlgebra

"""
Input sequences will be denoted by x
Output sequences will be denoted by y
The MPO transforming x to y will be denoted by W
X is a vector of data points x1, x2, x3 .... xN
Y is a vector of data points y1, y2, y3 .... yN
"""

function check_common_indices(X::Vector{MPS}, Y::Vector{MPS}, W::MPO)
    # Check if X and Y have the same number of data points
    if length(X) == length(Y)
        println("X and Y have same number of data points")
    else
        error("Number of Input and Output datapoints are different")
    end
    
    N = length(X) # Number of input data and also the output data
    Ind_sig_of_X = Array{Any}(undef, N); # List to contain a vector of sigma indices of x
    Ind_tau_of_Y = Array{Any}(undef, N); # List to contain a vector of tau indices of y
    for i in 1:N
        Ind_sig_of_X[i] = [inds(j, "x") for j in X[i]]
        Ind_tau_of_Y[i] = [inds(j, "y") for j in Y[i]]
    end
    
    if all(j->j==Ind_sig_of_X[1], Ind_sig_of_X)
        println("Input Data Points are properly constructed")
    else
        error("Input data points have different indices")
    end

    if all(j->j==Ind_tau_of_Y[1], Ind_tau_of_Y)
        println("Output Data Points are properly constructed")
    else
        error("Output data points have different indices")
    end

    Ind_sig_of_W = [inds(i, "x") for i in W]; # Vector of sigma indices of W
    Ind_tau_of_W = [inds(i, "y") for i in W]; # Vector of tau indices of W

    if Ind_sig_of_W == Ind_sig_of_X[1]
        println("W and x have a common index")
    else
        error("W and x dont have a common index")
    end

    if Ind_tau_of_W == Ind_tau_of_Y[1]
        println("W and y have a common index")
    else
        error("W and y dont have a common index")
    end

    return 0
end


"""
Computes inner product of two MPS x and y given in the form we are using
"""
function MPS_on_MPS(x::MPS, y::MPS)
    xc = deepcopy(x);
    yc = deepcopy(y);
    if x!=y
        xc[1] = order(xc[1]) == 3 ? xc[1]*delta(inds(xc[1], "bdary")) : xc[1];
        yc[1] = order(yc[1]) == 3 ? yc[1]*delta(inds(yc[1], "bdary")) : yc[1];
        xc[end] = order(xc[end]) == 3 ? xc[end]*delta(inds(xc[end], "bdary")) : xc[end];
        yc[end] = order(yc[end]) == 3 ? yc[end]*delta(inds(yc[end], "bdary")) : yc[end];
    end
    return inner(xc,yc)
end

"""
Computes the action of an MPO on an MPS
"""
function MPO_on_MPS(W::MPO, x::MPS)
    d1 = delta(inds(W[1], "bdary"), inds(x[1], "bdary"));
    d2 = delta(inds(W[end], "bdary"), inds(x[end], "bdary"));
    xc = deepcopy(x);
    xc[1] = xc[1]*d1;
    xc[end] = xc[end]*d2;
    z = applyMPO(W, xc);
    return z
end

"""
Computes <y-Wx|y-Wx> given x, y and w. This is the cost we are trying to minimize
"""
function diff_error(x::MPS, y::MPS, W::MPO)
    y_new = MPO_on_MPS(W, x);
    e1 = inner(y,y);
    e2 = inner(y_new, y_new);
    e3 = MPS_on_MPS(y, y_new);
    err = e1 + e2 - 2*e3;
    return e1, e2, e3, err
end



function A_B_f(x::MPS, W::MPO, l::Int64)
    L = length(x)
    xT = prime(x);
    WT = prime(W, "Wlink");
    WT = prime(WT, "x");
    #-------------------------------------------------------------------------------------
    # Constructing A, B and F tensors
    #-------------------------------------------------------------------------------------
    ind_W1 = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("Wlink, bdary"), inds(W[1])); # Boundary index for W[1]
    ind_a1 = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("xlink, bdary"), inds(x[1])); # Boundary index for x[1]
    A = delta(ind_a1, ind_W1, ind_a1', ind_W1');
    for k in 1:l-1
        A = A*x[k]*W[k]*WT[k]*xT[k];
    end

    ind_Wend = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("Wlink, bdary"), inds(W[end])); # Boundary index for W[end]
    ind_aend = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("xlink, bdary"), inds(x[end])); # Boundary index for x[end]
    B = delta(ind_aend, ind_Wend, ind_aend', ind_Wend');
    for k in reverse(l+1:L)
        B = B*x[k]*W[k]*WT[k]*xT[k];
    end

    f = A*B*x[l]*xT[l];
    #--------------------------------------------------------------------------------------
    return A, B, f
end


function Rt_Lt_U(x::MPS, y::MPS, W::MPO, l::Int64)
    L = length(x)
    #--------------------------------------------------------------------------------------
    # Constructing the Lt,Rt and U tensors
    #--------------------------------------------------------------------------------------
    
    ind_W1 = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("Wlink, bdary"), inds(W[1])); # Boundary index for W[1]
    ind_a1 = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("xlink, bdary"), inds(x[1])); # Boundary index for x[1]
    ind_c1 = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("ylink, bdary"), inds(y[1])); # Boundary index for y[1]
    Lt = delta(ind_a1, ind_W1, ind_c1);
    for k in 1:l-1
        Lt = Lt*x[k]*W[k]*y[k];
    end

    ind_Wend = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("Wlink, bdary"), inds(W[end])); # Boundary index for W[end]
    ind_aend = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("xlink, bdary"), inds(x[end])); # Boundary index for x[end]
    ind_cend = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("ylink, bdary"), inds(y[end])); # Boundary index for y[end]
    Rt = delta(ind_aend, ind_Wend, ind_cend);
    for k in reverse(l+1:L)
        Rt = Rt*x[k]*W[k]*y[k];
    end

    U = Lt*Rt*x[l]*y[l];

    return Rt, Lt, U
end


function C_D(W::MPO, l::Int64)
    L = length(W)
    #--------------------------------------------------------------------------------------
    # Constructing the C and D tensors
    #--------------------------------------------------------------------------------------
    Wt = prime(W, "Wlink");

    ind_W1 = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("Wlink, bdary"), inds(W[1])); # Boundary index for W[1]
    C = delta(ind_W1, ind_W1');
    for k in 1:l-1
        C = C*W[k]*Wt[k];
    end

    ind_Wend = filter(z->ITensors.dim(z) == 1 && tags(z) == TagSet("Wlink, bdary"), inds(W[end])); # Boundary index for W[end]
    D = delta(ind_Wend, ind_Wend');
    for k in reverse(l+1:L)
        D = D*W[k]*Wt[k];
    end

    return C, D
end


"""
This function optimizes W at index l through gradient descent
"""
function single_site_optimizer_one_step(X_data::Vector{MPS}, Y_data::Vector{MPS}, W::MPO, l::Int64, alpha::Float64, eta::Float64)::ITensor
    # X, Y are the input data and output data respectively
    # W is the MPO we need to optimize, l is the index to be optimized, alpha is the regularizer coeficient
    # eta is the stepsize of the update for W

    N = length(X_data);
    # Next form the tensors, A,B,C,D,F,L,R,U
    F = ITensor();
    U = ITensor();
    H = ITensor();

    for j in 1:N
        x = X_data[j]
        y = Y_data[j]

        a, b, f =  A_B_f(x, W, l);
        rt, lt, u = Rt_Lt_U(x, y, W, l);
        F = F + f;
        U = U + u;
    end
    c, d = C_D(W, l);
    H = c*d;
    
    # Form dCdW
    dCdW1 = noprime(F*W[l]);
    dCdW2 = noprime(U);
    dCdW3 = noprime(H*W[l]);
    dCdW = dCdW1 - dCdW2 + alpha*dCdW3;
    # Nudge W[l] towards the optimal value
    #println("Norm W[$l] ", norm(W[l]), " Norm grad ", norm(dCdW));
    W[l] = W[l] - eta*(norm(W[l])/norm(dCdW))*dCdW;
    #W[l] = W[l] - eta*(((W[l]*W[l])[])/((dCdW*dCdW)[]))*dCdW;
    #W[l] = W[l] - eta*dCdW;
    return W[l]
end

"""
This function runs one sweep
"""
function one_sweep(X::Vector{MPS}, Y::Vector{MPS}, W::MPO, alpha::Float64, eta::Float64)::MPO
    L = length(X[1])
    # Forward sweep
    for i in 1:L
        #println(i)
        W[i] = single_site_optimizer_one_step(X, Y, W, i, alpha, eta)
    end
    # Backward sweep
    for i in reverse(1:L-1)
        #println(i)
        W[i] = single_site_optimizer_one_step(X, Y, W, i, alpha, eta)
    end
    return W
end

function find_optimal_W(X::Vector{MPS}, Y::Vector{MPS}, D::Int64, alpha::Float64, eta::Float64, tolerance::Float64, max_steps::Int64, W_init::MPO)::Tuple{MPO, Float64}
    N_data = length(X);
    W = deepcopy(W_init);
    println("W initialized successfully")
    err = 1;
    steps = 0;
    while (err>tolerance && steps < max_steps)
        # Calculating error
        err = 0; e1 = 0; e2 = 0; e3 = 0;
        for i in 1:N_data 
            e1_new, e2_new, e3_new, err_new = diff_error(X[i], Y[i], W);
            err += err_new;
            e1 += e1_new; 
            e2 += e2_new;
            e3 += e3_new;
        end
        err = err + alpha*inner(W, W);
        println("Sweep no. ", steps, "   Error is ", e1, ' ', e2, ' ', e3, ' ', err, "  Norm of W is ", inner(W,W));
        W = one_sweep(X, Y, W, alpha, eta);
        steps = steps + 1;
    end
    return W, err
end



#---------------------------------------------------------------------------------------------------------------------------------
# PRE Linear optimizers
#---------------------------------------------------------------------------------------------------------------------------------
"""
This function optimizes W at index l according to the PRE paper
"""
function single_site_optimizer_PRE(X_data::Vector{MPS}, Y_data::Vector{MPS}, W::MPO, l::Int64, alpha::Float64)::ITensor
    # X, Y are the input data and output data respectively
    # W is the MPO we need to optimize, l is the index to be optimized, alpha is the regularizer coeficient

    N = length(X_data)
    # Next form the tensors, A,B,C,D,F,L,R,U
    F = ITensor();
    U = ITensor();
    H = ITensor();

    for j in 1:N
        x = X_data[j]
        y = Y_data[j]

        a, b, f =  A_B_f(x, W, l);
        rt, lt, u = Rt_Lt_U(x, y, W, l);

        F = F + f;
        U = U + u;
    end
    c, d = C_D(W, l);
    H = c*d;

    # Now reshape the tensors F + alpha*C*D and U into matrix and vector respectively
    sig_sig = randomITensor(inds(F, "x")); # Constructing additional tensor of ones for C*D
    sig_sig .*= 0.0;
    sig_sig .+= 1.0;

    G = F + alpha*H*sig_sig; # LHS of eqn A1
    idG = inds(G); # Indices of G
    dG = [dim(i) for i in idG]; # Dimensions of the indices of G
    G_arr = Array(G, idG[1], idG[3], idG[5], idG[2], idG[4], idG[6]); # Converting G to an Array
    G_mat = reshape(G_arr, (dG[1]*dG[3]*dG[5], dG[2]*dG[4]*dG[6])); # Converting G to a matrix

    idU = inds(U); # Indices of U which basically the RHS of eqn A1
    dU = [dim(i) for i in idU]; # Dimensions of the indices of U
    U_arr = Array(U, idU[1], idU[2], idU[3], idU[4]); # Converting U to an Array
    U_mat = reshape(U_arr, (dU[1]*dU[2]*dU[3], dU[4])); # Converting U to a matrix

    # Next find the various elements of W_idx by inverting F + alpha*C*D matrix and multiplying it with U
    if det(G_mat)==0 # Check that G_mat is non-singular
        return W[l]
    else
        W_mat = inv(G_mat)*U_mat; # New W[l] matrix

        W_arr = reshape(W_mat, (dU[1], dU[2], dU[3], dU[4])); # Converting the matrix W[l] into an Array
        # Reshape back the vector W into an appropriate tensor
        W_tensor = ITensor(W_arr, idU); # Conevrting the array W[l] to an appropriate tensor
        
        # Return W
        return W_tensor
    end
end


"""
This function runs one sweep
"""
function one_sweep_PRE(X::Vector{MPS}, Y::Vector{MPS}, W::MPO, alpha::Float64)::MPO
    L = length(X[1])
    # Forward sweep
    for i in 1:L
        #println(i)
        W[i] = single_site_optimizer_PRE(X, Y, W, i, alpha)
    end
    # Backward sweep
    for i in reverse(1:L-1)
        #println(i)
        W[i] = single_site_optimizer_PRE(X, Y, W, i, alpha)
    end
    return W
end

function find_optimal_W_PRE(X::Vector{MPS}, Y::Vector{MPS}, D::Int64, alpha::Float64, tolerance::Float64, max_steps::Int64, W_init::MPO)::Tuple{MPO, Float64}
    N_data = length(X);
    W = deepcopy(W_init);
    println("W initialized successfully")
    err = 1;
    steps = 0;
    while (err>tolerance && steps < max_steps)
        # Calculating error
        err = 0; e1 = 0; e2 = 0; e3 = 0;
        for i in 1:N_data
            e1_new, e2_new, e3_new, err_new = diff_error(X[i], Y[i], W);
            err += err_new;
            e1 += e1_new; 
            e2 += e2_new;
            e3 += e3_new;
        end
        err = err + alpha*inner(W, W);
        println("Sweep no. ", steps, "   Error is ", e1, ' ', e2, ' ', e3, ' ', err, "  Norm of W is ", inner(W,W));
        W = one_sweep_PRE(X, Y, W, alpha);
        steps = steps + 1;
    end
    return W, err
end
