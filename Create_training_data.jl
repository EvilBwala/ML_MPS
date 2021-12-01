using ITensors
using ITensors.HDF5
using Distributions
using NPZ
include("Data_gen_PRX.jl")

L = parse(Int, ARGS[1]) # Length of the patterns
sig_dims = parse(Int, ARGS[2]) # Dimensions of the spins = 2 for Michael's system
tau_dims = parse(Int, ARGS[3]) # Dimensions of the output = 2 (sinh(y) and cosh(y) in our case)
D = parse(Int, ARGS[4]) # Maximum Bond Dimension of the input (psi)
raw_datfile = ARGS[5] # Name of the file containing raw data
eta = parse(Float64, ARGS[6]); # Step size during optimizing the input data into psi
max_steps = parse(Int, ARGS[7]); # Maximum number of steps
tolerance = parse(Float64, ARGS[8]); # Tolerance during building the input psi

f = h5open("IO_Templates/xy_template.L$L.sig$sig_dims.tau$tau_dims.h5","r");
x_template = read(f, "x_template", MPS);
y_template = read(f, "y_template", MPS);
close(f);


#-------------------------------------------------------------------------------------------------------------------
# Initializing a random psi
psi_linkers = [min(Int(2^(i-1)), D) for i in 1:L/2+1];
psi_linkers = [psi_linkers[1:end-1] ; reverse(psi_linkers)];

ind_sig = Array{Index}(undef, L);
ind_a = Array{Index}(undef, L+1);

for i in 1:L+1
    if (i<L+1)  ind_sig[i] = inds(x_template[i], "x")[1]; end #Index for inputs same as that of x_template
    ind_a[i] = Index(psi_linkers[i], "xlink"); #Linker indices for X
    if (i == 1 || i == L+1)  ind_a[i] = addtags(ind_a[i], "bdary"); end # Boundary linkers have an extra tag
end
psi = MPS(L);
for i in 1:L  psi[i] = randomITensor(ind_sig[i], ind_a[i], ind_a[i+1]); end # psi is a MPS now
Z = inner(psi, psi);
psi = (1/sqrt(Z))*psi;
psi = orthogonalize(psi, 1);
println("psi initialized successfully");

#--------------------------------------------------------------------------------------------------------------------
raw_data = npzread(raw_datfile);
v_arr = trunc.(Int, raw_data["Pattern_list"]);
v_data = [v_arr[i,:] for i in 1:size(v_arr)[1]];

psi = find_optimal_psi(v_data, psi, D, eta, tolerance, max_steps);


#---------------------------------------------------------------------------
D_c = 1;
c_linkers = [min(Int(2^(i-1)), D_c) for i in 1:L/2+1];
c_linkers = [c_linkers[1:end-1] ; reverse(c_linkers)];

ind_tau = Array{Index}(undef, L);
ind_c = Array{Index}(undef, L+1);

for i in 1:L+1
    if i<L+1 ind_tau[i] = inds(y_template[i], "y")[1]; end #Index for outputs
    ind_c[i] = Index(c_linkers[i], "ylink"); #Linker indices for Y
    if (i == 1 || i == L+1) ind_c[i] = addtags(ind_c[i], "bdary"); end
end

y = MPS(L);

vy_data = raw_data["Protocol"]
vy_feature = [[sinh(i), cosh(i)] for i in vy_data];
for i in 1:10  y[i] = ITensor(vy_feature[i], ind_tau[i], ind_c[i], ind_c[i+1]); end

#-------------------------------------------------------------------------------------------------------------------
# Storing the training data in the training data folder
#--------------------------------------------------------------------------------------------------------------------
datafolder="Training_Data"

if isdir(datafolder)
    println("$datafolder already exists");
else
    mkdir(datafolder);
end

f = h5open("$datafolder/$raw_datfile.L$L.sig$sig_dims.tau$tau_dims.h5","w")
write(f,"psi",psi, "y", y);
close(f);