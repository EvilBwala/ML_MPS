using ITensors
using LinearAlgebra
include("MPS_functions.jl")

L = 10;
sig_dims = 2;
tau_dims = 3;
D = 1; # Bond Dimensions
D_a = 2;
D_w = 10;
D_c = 1;

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

y_pred = MPO_on_MPS(W, x);

function P_Q(y1::MPS, y2::MPS, l::Int64)
    ind_a1 = filter(z->ITensors.dim(z) == 1 && hastags(z, "bdary"), inds(y1[1])); # Boundary index for y1[1]
    ind_c1 = filter(z->ITensors.dim(z) == 1 && hastags(z, "bdary"), inds(y2[1])); # Boundary index for y2[1]
    P = delta(ind_a1, ind_c1);
    for k in 1:l-1
        P = P*y1[k]*y2[k];
    end

    ind_aend = filter(z->ITensors.dim(z) == 1 && hastags(z, "bdary"), inds(y1[end])); # Boundary index for y1[end]
    ind_cend = filter(z->ITensors.dim(z) == 1 && hastags(z, "bdary"), inds(y2[end])); # Boundary index for y2[end]
    Q = delta(ind_aend, ind_cend);
    for k in reverse(l+1:L)
        Q = Q*y1[k]*y2[k];
    end
    return P, Q
end

function single_site_optimizer_y_signal(y_signal::MPS, y_pred::MPS, l::Int64)
    A, B = P_Q(y_signal, prime(y_signal, "ylink"), l);
    C, D = P_Q(y_signal, y_pred, l);
    F = A*B;
    G = C*D*y_pred[l];
    if F[1,1,1,1] != 0
        return (1/F[1,1,1,1])*G;
    else
        return y_signal[l]
    end
end

function one_sweep_y_signal(y_signal::MPS, y_pred::MPS)
    L = length(y_signal);
    for l in 1:L
        y_signal[l] = single_site_optimizer_y_signal(y_signal, y_pred, l);
        #println(l)
    end
    for l in reverse(1:L-1)
        y_signal[l] = single_site_optimizer_y_signal(y_signal, y_pred, l);
        #println(l)
    end
    return y_signal
end

function optimize_y_signal(y_signal::MPS, y_pred::MPS, tolerance::Float64, max_steps::Int64)
    err = 1;
    steps = 0;
    while err>tolerance && steps<max_steps
        err = inner(y_signal, y_signal) + inner(y_pred, y_pred) - 2*MPS_on_MPS(y_signal, y_pred);
        println("Sweep no. ", steps, "   Error is ", inner(y_signal, y_signal), ' ', inner(y_pred, y_pred), ' ', MPS_on_MPS(y_signal, y_pred), ' ', err);
        y_signal = one_sweep_y_signal(y_signal, y_pred);
        steps = steps + 1;
    end
    return y_signal, err
end