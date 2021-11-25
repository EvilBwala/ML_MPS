using ITensors
using LinearAlgebra
using Distributions
using Random

include("MPS_functions.jl")

L = 10;
b = Binomial();
v_data = [rand(b, L) .+ 1 for i in 1:10];


L = length(v_data[1]);

D = 10;
sig_dims = 2;
W_linkers = [min(Int(2^(i-1)), D) for i in 1:L/2+1];
psi_linkers = [W_linkers[1:end-1] ; reverse(W_linkers)];

ind_sig = Array{Index}(undef, L);
ind_a = Array{Index}(undef, L+1);
for i in 1:L+1
    if (i<L+1)  ind_sig[i] = Index(sig_dims, "x"); end #Index for inputs
    ind_a[i] = Index(psi_linkers[i], "xlink"); #Linker indices for X
    if (i == 1 || i == L+1)  ind_a[i] = addtags(ind_a[i], "bdary"); end # Boundary linkers have an extra tag
end
psi = MPS(L);
for i in 1:L
    psi[i] = ITensor(ind_sig[i], ind_a[i], ind_a[i+1]); # psi is a MPS now
    psi[i] = onehot(ind_sig[i]=>1, ind_a[i]=>1, ind_a[i+1]=>1)
end
Z = inner(psi, psi);
psi = (1/sqrt(Z))*psi;


# v_data is the data here and we need to optimize x

function psi_v(psi::MPS, v::Vector)
    pd = delta(inds(psi[1], "bdary, xlink"), inds(psi[end], "bdary, xlink"));
    L = length(psi);
    for i in 1:L
        T = setelt(inds(psi[i], "x")[1] => v[i]);
        pd = pd*psi[i]*T;
    end
    return pd
end

function psi_prime_v(psi::MPS, k1::Int64, k2::Int64, v::Vector)
    pd = delta(inds(psi[1], "bdary, xlink"), inds(psi[end], "bdary, xlink"));
    L = length(psi);
    for i in 1:L
        if(i==k1 || i==k2)
            T = setelt(inds(psi[i], "x")[1] => v[i]);
            pd = pd*T;
            continue
        end
        T = setelt(inds(psi[i], "x")[1] => v[i]);
        pd = pd*psi[i]*T;
    end
    return pd
end

"""
xT = prime(x, "xlink")
pd = delta(inds(x[1], "bdary"), inds(xT[1], "bdary"))
for i in 1:L
    pd = pd*x[i]*xT[i];
end
# This is equivalent to
pd1 = inner(x, x)
"""
"""
x = orthogonalize(x, 4);
xT = prime(x, "xlink");
pd = delta(inds(x[1], "bdary"), inds(xT[1], "bdary"))*delta(inds(x[end], "bdary"), inds(xT[end], "bdary"))
for i in 1:L
    if i==4 || i==5
        continue
    end
    pd = pd*x[i]*xT[i];
end

# Code for gradient descent at a specific k for one training data
k = 9;
x = orthogonalize(x, k);
v = [1,2,1,1,2,1,2,1,2,2];
N_data = 1;
eta = 0.01;

A = x[k]*x[k+1];
Z = inner(x,x);
dLdA = 2*A/Z - (2/N_data)*psi_prime_v(x, k, v)/(psi_v(x, v)[]);
A = A - eta*dLdA;
U,S,V = svd(A, uniqueinds(x[k], x[k+1]), cutoff = 1E-3);

x[k] = replacetags(U, "Link, u", "xlink");
x[k+1] = replacetags(S*V, "Link, v" => "xlink",  "Link, u" => "xlink");
"""

# This function assumes that psi is orthogonalized at site k
# D is the maximum bond Dimensions
# eta is the step size, direction can be either "forward" or "backward"
function single_site_one_step(psi::MPS, k::Int64, D::Int64, v_data::Vector{Vector{Int64}}, eta::Float64, direction::String)
    N_data = length(v_data); # Number of Data points
    L = length(psi); # Length of MPS
    nxt_ind = (direction=="forward") ? 1 : -1; # Check direction of optimization
    if(k+nxt_ind<1 || k+nxt_ind>L)
        error("Next index is out of bounds")
    end
    A = psi[k]*psi[k+nxt_ind]; # Form the two-site MPS
    Z = inner(psi, psi); # Find the inner product Z
    dLdA_1 = 2*A/Z; # Form Z'/Z under the assumption that psi is in mixed canonical form
    #---------------------------------------------------------------------------------------
    # Form the second term in eqn B2 of the PRX paper
    #---------------------------------------------------------------------------------------
    dLdA_2 = ITensor(inds(A));
    for v in v_data
        dLdA_2 .+= psi_prime_v(psi, k, k+nxt_ind, v)/(psi_v(psi, v)[]);
    end
    #---------------------------------------------------------------------------------------
    dLdA = dLdA_1 - (2/N_data)*dLdA_2;
    #println(norm(A), ' ', norm(dLdA));
    #eta = 0.1*norm(A)/norm(dLdA);
    A_new = A - eta*dLdA;
    U, S, V = ITensors.svd(A_new, uniqueinds(psi[k], psi[k+nxt_ind]), maxdim=D); # Perform svd
    psi[k] = replacetags(U, "Link, u" => "xlink");
    psi[k+nxt_ind] = replacetags(S*V, "Link, v" => "xlink",  "Link, u" => "xlink");
    return psi
end

# This assumes that psi is orthogonalized at site 1, the first site
function one_sweep_for_psi(psi::MPS, D::Int64, v_data::Vector{Vector{Int64}}, eta::Float64)
    L = length(psi)
    #-------------------------------------------------------------------
    # Forward sweep
    for k in 1:L-1
        psi = single_site_one_step(psi, k, D, v_data, eta, "forward");
    end
    #-------------------------------------------------------------------
    # Backward sweep
    for k in reverse(2:L)
        psi = single_site_one_step(psi, k, D, v_data, eta, "backward");
    end
    return psi
end

function find_optimal_psi(v_data::Vector{Vector{Int64}}, dim_of_spins::Int64, D::Int64, eta::Float64, tolerance::Float64, max_steps::Int64)
    #--------------------------------------------------------------------------------------------------------------
    # Creating a random psi
    #--------------------------------------------------------------------------------------------------------------
    N_data = length(v_data);
    L = length(v_data[1]);

    sig_dims = dim_of_spins;
    W_linkers = [min(Int(2^(i-1)), D) for i in 1:L/2+1];
    psi_linkers = [W_linkers[1:end-1] ; reverse(W_linkers)];

    ind_sig = Array{Index}(undef, L);
    ind_a = Array{Index}(undef, L+1);
    for i in 1:L+1
        if (i<L+1)  ind_sig[i] = Index(sig_dims, "x"); end #Index for inputs
        ind_a[i] = Index(psi_linkers[i], "xlink"); #Linker indices for X
        if (i == 1 || i == L+1)  ind_a[i] = addtags(ind_a[i], "bdary"); end # Boundary linkers have an extra tag
    end
    psi = MPS(L);
    for i in 1:L
        psi[i] = randomITensor(ind_sig[i], ind_a[i], ind_a[i+1]); # psi is a MPS now
        #psi[i] = onehot(ind_sig[i]=>1, ind_a[i]=>1, ind_a[i+1]=>1)
    end
    Z = inner(psi, psi);
    psi = (1/sqrt(Z))*psi;
    psi = orthogonalize(psi, 1);
    #-----------------------------------------------------------------------------------------------------------------
    # Random psi with appropriate indices created
    #-----------------------------------------------------------------------------------------------------------------

    err = 1;
    steps = 0;
    while (abs(err)>tolerance && steps<max_steps)
        global psi
        s = [(psi_v(psi, v)[])/sqrt(inner(psi, psi)) for v in v_data];
        lkhood = [-log(i*i) for i in s];
        err = sum(lkhood)/N_data;
        psi = one_sweep_for_psi(psi, D, v_data, eta);
        steps += 1;
        println(steps, ' ', abs(err), ' ', inner(psi, psi))
    end
    return psi
end


#-----------------------------------------------------------------------------------------------------------------------
# Calculating Shannon entropy of a dataset
#-----------------------------------------------------------------------------------------------------------------------



