using ITensors
using LinearAlgebra
using ITensors.HDF5

L = parse(Int, ARGS[1])
sig_dims = parse(Int, ARGS[2])
tau_dims = parse(Int, ARGS[3])
D_a = 10;
D_c = 1;


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

if isdir("IO_Templates")
    println("IO_Templates folder already exists");
else
    mkdir("IO_Templates");
end

if isfile("IO_Templates/xy_template.L$L.sig$sig_dims.tau$tau_dims.h5")
    println("Template already exists");
else
    f = h5open("IO_Templates/xy_template.L$L.sig$sig_dims.tau$tau_dims.h5","w")
    write(f,"x_template",x, "y_template", y)
    close(f)
end
