using ITensors
using LinearAlgebra

L = 10;
sig_dims = 2;
tau_dims = 3;
D = 1; # Bond Dimensions
D_w = 10;
W_linkers = [min(Int(2^(i-1)), D_w) for i in 1:L/2+1];
W_linkers = [W_linkers[1:end-1] ; reverse(W_linkers)];

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
    
    ind_a[i] = Index(D, "xlink"); #Linker indices for X
    ind_b[i] = Index(W_linkers[i], "Wlink"); #Linker indices for W
    ind_c[i] = Index(D, "ylink"); #Linker indices for Y

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
X = [x, 2*x]
Y = [y, 2*y]


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

