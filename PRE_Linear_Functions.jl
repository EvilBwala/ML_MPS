"""
This function optimizes W at index l
"""
function single_site_optimizer(X_data::Vector{MPS}, Y_data::Vector{MPS}, W::MPO, l::Int64, alpha::Float64)::ITensor
    # X, Y are the input data and output data respectively
    # W is the MPO we need to optimize, l is the index to be optimized, alpha is the regularizer coeficient
    

    N = length(X_data)
    # Next form the tensors, A,B,C,D,F,L,R,U
    A = ITensor();
    B = ITensor();
    C = ITensor();
    D = ITensor();
    F = ITensor();
    Lt = ITensor();
    Rt = ITensor();
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
function one_sweep(X::Vector{MPS}, Y::Vector{MPS}, W::MPO, alpha::Float64)::MPO
    L = length(X[1])
    # Forward sweep
    for i in 1:L
        #println(i)
        W[i] = single_site_optimizer(X, Y, W, i, alpha)
    end
    # Backward sweep
    for i in reverse(1:L-1)
        #println(i)
        W[i] = single_site_optimizer(X, Y, W, i, alpha)
    end
    return W
end

function find_optimal_W(X::Vector{MPS}, Y::Vector{MPS}, D::Int64, alpha::Float64, tolerance::Float64, max_steps::Int64)::MPO
    N_data = length(X);
    L = length(X[1]);
    # First check if elements x and w, w and y have matching indices
    #check_common_indices(X, Y, W)

    W_linkers = [min(Int(2^(i-1)), D) for i in 1:L/2+1];
    W_linkers = [W_linkers[1:end-1] ; reverse(W_linkers)];
    ind_b = Array{Index}(undef, L+1);
    
    #-----------------------------------------------------------------------
    #Building indices for MPO W
    #-----------------------------------------------------------------------
    for i in 1:L+1
        ind_b[i] = Index(W_linkers[i], "Wlink"); #Linker indices for W
        if (i == 1 || i == L+1) ind_b[i] = addtags(ind_b[i], "bdary"); end
    end
    # Initializing w
    W = MPO(L);
    for i in 1:L W[i] = randomITensor(inds(X[1][i], "x")[1], inds(Y[1][i], "y"), ind_b[i], ind_b[i+1]); end # W is a MPO now
    W = (1/sqrt(inner(W,W)))*W;
    W = orthogonalize(W, 1);
    println("W initialized successfully")
    err = 1;
    steps = 0;
    while (err>tolerance && steps < max_steps)
        W = one_sweep(X, Y, W, alpha);
        # Calculating error
        err = 0;
        for i in 1:N_data err += diff_error(X[i], Y[i], W); end
        err = err + alpha*inner(W, W);
        println(steps, ' ', err);
        steps = steps + 1;
    end
    return W
end
