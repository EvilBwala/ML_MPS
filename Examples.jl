"""
IMPORTANT NOTE: 
Distributions package of julia contains the method dim which conflicts with the usage of dim in MPS_functions.jl
So the examples need to be run independently.
"""


"""
using ITensors
using LinearAlgebra
include("MPS_functions.jl")

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

"""
#---------------------------------------------------------------------------------------------------------------------------------------


using ITensors
using LinearAlgebra
using Distributions
include("Data_gen_PRX.jl")


L = 10;
b = Binomial();
v_data = [rand(b, L) .+ 1 for i in 1:100];
dim_of_spins = 2;
D = 20;
eta = 0.01;
max_steps = 1000;
tolerance = 1E-3;
psi = find_optimal_psi(v_data, dim_of_spins, D, eta, tolerance, max_steps)