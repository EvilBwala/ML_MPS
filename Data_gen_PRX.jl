"""
This file contains functions for genrating psi as mentioned in the PRX paper
"""

"""
Function to find psi(v) given a MPS psi and a data vector v. 
v is of the form [1,2,1,1,2,....., L such labels] 1,2...d is the dimensionality of the spin
For Michael's system d = 2, so the labels can be either 1 or 2
"""
function psi_v(psi::MPS, v::Vector)
    pd = delta(inds(psi[1], "bdary, xlink"), inds(psi[end], "bdary, xlink"));
    L = length(psi);
    for i in 1:L
        T = setelt(inds(psi[i], "x")[1] => v[i]);
        pd = pd*psi[i]*T;
    end
    return pd
end

"""
This function constructs psi_prime(v) as mentioned in the PRX paper
"""
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
This function assumes that psi is orthogonalized at site k
D is the maximum bond Dimensions
eta is the step size, direction can be either "forward" or "backward"
"""
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

"""
This assumes that psi is orthogonalized at site 1, the first site
"""
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


"""
This function calculates the Shannon entropy of the dataset v_data
"""

function shannon_ent(v_data::Vector{Vector{Int64}})
    data_freq = [];
    v_datac = copy(v_data)
    L = length(v_datac)
    while v_datac!=[]
        deleteat!(v_datac, findall(x->x==v_datac[1], v_datac));
        push!(data_freq, L - length(v_datac));
        L = length(v_datac);
    end
    prb = (1/sum(data_freq))*data_freq;
    Sentropy = -sum(prb .* log.(prb));
    return Sentropy
end


"""
This function constructs the optimal psi given data in the form of a vector of vectors
"""

function find_optimal_psi(v_data::Vector{Vector{Int64}}, psi::MPS, D::Int64, eta::Float64, tolerance::Float64, max_steps::Int64)
    N_data = length(v_data);
    """
    #--------------------------------------------------------------------------------------------------------------
    # Creating a random psi
    #--------------------------------------------------------------------------------------------------------------
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
    println("psi initialized successfully");
    #-----------------------------------------------------------------------------------------------------------------
    # Random psi with appropriate indices created
    #-----------------------------------------------------------------------------------------------------------------
    """
    err = 1;
    s_ent = shannon_ent(v_data);
    println("Shannon entropy of data is ", s_ent)
    steps = 0;
    while (abs(err)>tolerance && steps<max_steps)
        s = [(psi_v(psi, v)[])/sqrt(inner(psi, psi)) for v in v_data];
        lkhood = [-log(i*i) for i in s];
        err = s_ent - sum(lkhood)/N_data;
        psi = one_sweep_for_psi(psi, D, v_data, eta);
        steps += 1;
        println("Sweep no.", steps, "  Error is ", abs(err))
    end
    return psi
end

#-------------------------------------------------------------------------------------------------------------------------
# Create simple training data 
#-------------------------------------------------------------------------------------------------------------------------

function simple_training_data(x_template::MPS, y_template::MPS, vx_data::Union{Matrix, Vector}, vy_data::Union{Matrix, Vector})::Tuple{MPS, MPS}
    L = length(x_template);
    #-------------------------------------------------------------------------------------------------------------------
    # Creating indices for MPS x and y
    D_a = 1
    a_linkers = [min(Int(2^(i-1)), D_a) for i in 1:L/2+1];
    a_linkers = [a_linkers[1:end-1] ; reverse(a_linkers)];
    D_c = 1;
    c_linkers = [min(Int(2^(i-1)), D_c) for i in 1:L/2+1];
    c_linkers = [c_linkers[1:end-1] ; reverse(c_linkers)];

    ind_sig = Array{Index}(undef, L);
    ind_a = Array{Index}(undef, L+1);
    ind_tau = Array{Index}(undef, L);
    ind_c = Array{Index}(undef, L+1);

    for i in 1:L+1
        if (i<L+1)  ind_sig[i] = inds(x_template[i], "x")[1]; end #Index for inputs same as that of x_template
        ind_a[i] = Index(a_linkers[i], "xlink"); #Linker indices for X
        if (i == 1 || i == L+1)  ind_a[i] = addtags(ind_a[i], "bdary"); end # Boundary linkers have an extra tag
        
        if i<L+1 ind_tau[i] = inds(y_template[i], "y")[1]; end #Index for outputs
        ind_c[i] = Index(c_linkers[i], "ylink"); #Linker indices for Y
        if (i == 1 || i == L+1) ind_c[i] = addtags(ind_c[i], "bdary"); end
    end

    y = MPS(L);
    vy_feature = [[tanh(i), sech(i)] for i in vy_data];
    for i in 1:L  y[i] = ITensor(vy_feature[i], ind_tau[i], ind_c[i], ind_c[i+1]); end

    x = MPS(L);
    vx_feature = [zeros(1,sig_dims) for i in vx_data];
    for i in 1:L
        vx_feature[i][vx_data[i]] = 1;
        x[i] = ITensor(vx_feature[i], ind_sig[i], ind_a[i], ind_a[i+1]);
    end

    return x, y
end


    
    
