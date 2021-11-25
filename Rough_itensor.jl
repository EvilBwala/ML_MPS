using ITensors
using LinearAlgebra
include("MPS_functions.jl")

L = 10;
sig_dims = 2;
tau_dims = 3;
D = 1; # Bond Dimensions
D_a = 2;
D_w = 10;
D_c = 2;

a_linkers = [min(Int(2^(i-1)), D_a) for i in 1:L/2+1];
a_linkers = [a_linkers[1:end-1] ; reverse(a_linkers)];

W_linkers = [min(Int(2^(i-1)), D_w) for i in 1:L/2+1];
W_linkers = [W_linkers[1:end-1] ; reverse(W_linkers)];

c_linkers = [min(Int(2^(i-1)), D_c) for i in 1:L/2+1];
c_linkers = [c_linkers[1:end-1] ; reverse(c_linkers)];


ind_sig = Array{Index}(undef, L);
ind_a = Array{Index}(undef, L+1);
ind_tau = Array{Index}(undef, L);
ind_b = Array{Index}(undef, L+1);
ind_c = Array{Index}(undef, L+1);

for i in 1:L+1
    if i<L+1
        ind_sig[i] = Index(sig_dims, "x"); #Index for inputs
        ind_tau[i] = Index(tau_dims, "y"); #Index for outputs
    end
    
    ind_a[i] = Index(a_linkers[i], "xlink"); #Linker indices for X
    ind_b[i] = Index(W_linkers[i], "Wlink"); #Linker indices for W
    ind_c[i] = Index(c_linkers[i], "ylink"); #Linker indices for Y

    if (i == 1 || i == L+1)
        ind_a[i] = addtags(ind_a[i], "bdary");
        ind_b[i] = addtags(ind_b[i], "bdary");
        ind_c[i] = addtags(ind_c[i], "bdary"); 
    end
end

x = MPS(L);
y = MPS(L);
W = MPO(L);

for i in 1:L
    x[i] = randomITensor(ind_sig[i], ind_a[i], ind_a[i+1]); # x is a MPS now
    y[i] = randomITensor(ind_tau[i], ind_c[i], ind_c[i+1]); # y is a MPS now
    W[i] = randomITensor(ind_sig[i], ind_tau[i], ind_b[i], ind_b[i+1]); # W is a MPO now
end
x = (1/sqrt(inner(x,x)))*x;
y = (1/sqrt(inner(y,y)))*y;
W = (1/sqrt(inner(W,W)))*W;

X = [x, 2*x];
Y = [y, 2*y];



#----------------------------------------------------------------------------------------------------
# All down below will become a part of MPS_functions.jl

sig_ind = [ind(i, 1) for i in x];
a_ind = push!([ind(i, 2) for i in x] , ind(x[end], 3));
W_sig_ind = [ind(i, 1) for i in W];
W_tau_ind = [ind(i, 2) for i in W];
W_b_ind = push!([ind(i, 3) for i in W] , ind(W[end], 4));
c_ind = push!([ind(i, 2) for i in y] , ind(y[end], 3));


xT = prime(x);
WT = prime(W, W_b_ind);
WT = prime(WT, W_sig_ind);

l = 4;

#-------------------------------------------------------------------------------------
# Constructing A, B and F tensors
#-------------------------------------------------------------------------------------
A = delta(a_ind[1], W_b_ind[1], a_ind[1]', W_b_ind[1]');

for k in 1:l-1
    A = A*x[k]*W[k]*WT[k]*xT[k];
end

B = delta(a_ind[end], W_b_ind[end], a_ind[end]', W_b_ind[end]');
for k in reverse(l+1:L)
    B = B*x[k]*W[k]*WT[k]*xT[k];
end

F = A*B*x[l]*xT[l];
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
# Constructing the Lt,Rt and U tensors
#--------------------------------------------------------------------------------------
Lt = delta(a_ind[1], W_b_ind[1], c_ind[1]);
for k in 1:l-1
    Lt = Lt*x[k]*W[k]*y[k];
end

Rt = delta(a_ind[end], W_b_ind[end], c_ind[end]);
for k in reverse(l+1:L)
    Rt = Rt*x[k]*W[k]*y[k];
end

U = Lt*Rt*x[l]*y[l];
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
# Constructing the C and D tensors
#--------------------------------------------------------------------------------------
Wt = prime(W, W_b_ind);

C = delta(W_b_ind[1], W_b_ind[1]');
for k in 1:l-1
    C = C*W[k]*Wt[k];
end

D = delta(W_b_ind[end], W_b_ind[end]');
for k in reverse(l+1:L)
    D = D*W[k]*Wt[k];
end

#-------------------------------------------------------------------------------------
# Reshaping F, CD, U to find W
#-------------------------------------------------------------------------------------

sig_sig = randomITensor(inds(F, "x")); # Constructing additional tensor of ones for C*D
sig_sig .*= 0.0;
sig_sig .+= 1.0;

G = F + C*D*sig_sig; # LHS of eqn A1
idG = inds(G); # Indices of G
dG = [dim(i) for i in idG]; # Dimensions of the indices of G
G_arr = Array(G, idG[1], idG[3], idG[5], idG[2], idG[4], idG[6]); # Converting G to an Array
G_mat = reshape(G_arr, (dG[1]*dG[3]*dG[5], dG[2]*dG[4]*dG[6])); # Converting G to a matrix


idU = inds(U); # Indices of U which basically the RHS of eqn A1
dU = [dim(i) for i in idU]; # Dimensions of the indices of U
U_arr = Array(U, idU[1], idU[2], idU[3], idU[4]); # Converting U to an Array
U_mat = reshape(U_arr, (dU[1]*dU[2]*dU[3], dU[4])); # Converting U to a matrix

W_mat = inv(G_mat)*U_mat; # New W[l] matrix

W_arr = reshape(W_mat, (dU[1], dU[2], dU[3], dU[4])); # Converting the matrix W[l] into an Array

W_tensor = ITensor(W_arr, idU); # Conevrting the array W[l] to an appropriate tensor

"""

for s in 1:4
    global W
    W = one_sweep(X, Y, W, 0.01);
end

"""

v = [1,2,1,1,2,1,2,1,2,2]

pd = delta(ind_a[1], ind_a[end])
for i in 1:L
    if i == 6 || i == 7
        continue
    end
    T = setelt(inds(x[i], "x")[1] => v[i]);
    pd = pd*T*x[i];
end


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

function MPO_on_MPS(W::MPO, x::MPS)
    d1 = delta(inds(W[1], "bdary"), inds(x[1], "bdary"));
    d2 = delta(inds(W[end], "bdary"), inds(x[end], "bdary"));
    xc = deepcopy(x);
    xc[1] = xc[1]*d1;
    xc[end] = xc[end]*d2;
    z = applyMPO(W, xc);
    return z
end

function diff_error(x::MPS, y::MPS, W::MPO)
    y_new = MPO_on_MPS(W, x);
    e1 = inner(y,y);
    e2 = inner(y_new, y_new);
    e3 = MPS_on_MPS(y, y_new);
    err = e1 + e2 - 2*e3;
    return err
end



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
    W[l] = W[l] - eta*dCdW;
    return W[l]
end

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

function find_optimal_W(X::Vector{MPS}, Y::Vector{MPS}, D::Int64, alpha::Float64, eta::Float64, tolerance::Float64, max_steps::Int64)::MPO
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
        W = one_sweep(X, Y, W, alpha, eta);
        # Calculating error
        err = 0;
        for i in 1:N_data err += diff_error(X[i], Y[i], W); end
        err = err + alpha*inner(W, W);
        println(steps, ' ', err);
        steps = steps + 1;
    end
    return W
end