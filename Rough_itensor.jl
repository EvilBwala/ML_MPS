using ITensors

L = 10
sig_dims = 2
tau_dims = 2
D = 1 # Bond Dimensions
D_w = 10
W_linkers = [min(Int(2^(i-1)), D_w) for i in 1:L/2+1]
W_linkers = [W_linkers[1:end-1] ; reverse(W_linkers)]

ind_sig = Array{Index}(undef, L)
ind_a = Array{Index}(undef, L+1)
ind_tau = Array{Index}(undef, L)
ind_b = Array{Index}(undef, L+1)

for i in 1:L+1
    if i<L+1
        ind_sig[i] = Index(sig_dims, "x") #Index for inputs
        ind_tau[i] = Index(tau_dims, "y") #Index for outputs
    end
    
    ind_a[i] = Index(D) #Linker indices for X
    ind_b[i] = Index(W_linkers[i]) #Linker indices for W
end

x = MPS(L);
W = MPO(L);

for i in 1:L
    x[i] = randomITensor(ind_sig[i], ind_a[i], ind_a[i+1]) # x is a MPS now
    W[i] = randomITensor(ind_sig[i], ind_tau[i], ind_b[i], ind_b[i+1]) # W is a MPO now
end
