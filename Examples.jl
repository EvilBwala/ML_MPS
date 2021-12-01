"""
IMPORTANT NOTE: 
Distributions package of julia contains the method dim which conflicts with the usage of dim in MPS_functions.jl
So the examples need to be run independently.
"""



using ITensors
using LinearAlgebra
using Distributions
include("MPS_functions.jl")
include("Data_gen_PRX.jl")


#---------------------------------------------------------------------------------------------------------------------------------------
L = 10;
sig_dims = 2;
tau_dims = 3;
D_a = 2;
D_c = 2;

a_linkers = [min(Int(2^(i-1)), D_a) for i in 1:L/2+1];
a_linkers = [a_linkers[1:end-1] ; reverse(a_linkers)];

c_linkers = [min(Int(2^(i-1)), D_c) for i in 1:L/2+1];
c_linkers = [c_linkers[1:end-1] ; reverse(c_linkers)];


ind_sig = Array{Index}(undef, L);
ind_a = Array{Index}(undef, L+1);
ind_tau = Array{Index}(undef, L);
ind_c = Array{Index}(undef, L+1);

for i in 1:L+1
    if i<L+1
        ind_sig[i] = Index(sig_dims, "x"); #Index for inputs
        ind_tau[i] = Index(tau_dims, "y"); #Index for outputs
    end
    
    ind_a[i] = Index(a_linkers[i], "xlink"); #Linker indices for X
    ind_c[i] = Index(c_linkers[i], "ylink"); #Linker indices for Y

    if (i == 1 || i == L+1)
        ind_a[i] = addtags(ind_a[i], "bdary");
        ind_c[i] = addtags(ind_c[i], "bdary"); 
    end
end

x = MPS(L);
y = MPS(L);

for i in 1:L
    x[i] = randomITensor(ind_sig[i], ind_a[i], ind_a[i+1]); # x is a MPS now
    y[i] = randomITensor(ind_tau[i], ind_c[i], ind_c[i+1]); # y is a MPS now
end
x = (1/sqrt(inner(x,x)))*x;
y = (1/sqrt(inner(y,y)))*y;

X = [x, 2*x];
Y = [y, 2*y];

D = 10;
alpha = 0.01;
eta = 0.2;
tolerance = 0.01;
max_steps = 100;
W = find_optimal_W(X, Y, D, alpha, eta, tolerance, max_steps)

#---------------------------------------------------------------------------------------------------------------------------------------

L = 10
sig_dims = 2
tau_dims = 2
D = 20

f = h5open("IO_Templates/xy_template.L$L.sig$sig_dims.tau$tau_dims.h5","r");
x_template = read(f, "x_template", MPS);
y_template = read(f, "y_template", MPS);


W_linkers = [min(Int(2^(i-1)), D) for i in 1:L/2+1];
psi_linkers = [W_linkers[1:end-1] ; reverse(W_linkers)];

ind_sig = Array{Index}(undef, L);
ind_a = Array{Index}(undef, L+1);

for i in 1:L+1
    if (i<L+1)  ind_sig[i] = inds(x_template[i], "x")[1]; end #Index for inputs same as that of x_template
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


b = Binomial();
v_data = [rand(b, L) .+ 1 for i in 1:100];
eta = 0.01;
max_steps = 100;
tolerance = 1E-3;
psi = find_optimal_psi(v_data, psi, D, eta, tolerance, max_steps);