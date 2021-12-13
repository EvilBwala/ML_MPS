
"""
The code creates the Optimal MPO, W given the training data in appropriate format.
All the training data needs to be stored in a folder. This code reads all the training data
from this folder and constructs the optimal W using functions in MPS_functions.jl
INPUT:
ARGS[1] = Name of folder where training data is stored
ARGS[2] = D is the maximum bond dimension of the optimal W to be built
ARGS[3] = alpha is the regularizer for the MPO
ARGS[4] = eta is the step size for the gradient descent algorithm
ARGS[5] = tolerance is the tolerance of error for the optimal w
ARGS[6] = max_steps1 is the total number of steps we want our find_optimal_W algorithm to run. This is stepwise gradient descent
ARGS[7] = max_steps2 is the total number of steps we want our find_optimal_W_PRE algorithm to run. This is finding W from arrangement and inversion
ARGS[8] = ncycles is the number of cycles of alternating between the two algorithms
OUTPUT:
The optimal W which is created in stored in a folder named Optimal_W_MPOs under the name
Optimal_W.h5.
"""

using ITensors
using ITensors.HDF5
using NPZ
include("MPS_functions.jl")
include("Data_gen_PRX.jl")

raw_datafolder = ARGS[1];
L = parse(Int, ARGS[2]);
sig_dims = parse(Int, ARGS[3]);
tau_dims = parse(Int, ARGS[4]);
num_trials = parse(Int, ARGS[5]);
training_datafolder = ARGS[6];
training_datafile = ARGS[7];
D = parse(Int, ARGS[8]);
alpha = parse(Float64, ARGS[9]);
eta = parse(Float64, ARGS[10]);
tolerance = parse(Float64, ARGS[11]);
max_steps1 = parse(Int, ARGS[12]);
max_steps2 = parse(Int, ARGS[13]);
ncycles = parse(Int, ARGS[14]);
Optimal_MPO_name = ARGS[15];

f = h5open("IO_Templates/xy_template.L$L.sig$sig_dims.tau$tau_dims.h5","r");
x_template = read(f, "x_template", MPS);
y_template = read(f, "y_template", MPS);
close(f);

raw_data_files = filter(z->occursin("npz", z) , readdir(raw_datafolder));

X_data = Vector{MPS}();
Y_data = Vector{MPS}();


for raw_data_file in raw_data_files
    raw_data = npzread("$raw_datafolder/$raw_data_file");
    v_arr = trunc.(Int, raw_data["Pattern_list"]);
    v_data = [v_arr[i,1:L] for i in 1:size(v_arr)[1]];
    for k in 1:num_trials
        println("$raw_data_file Trial no. ", k);
        vy_data = raw_data["Protocol"][1:L];
        vx_data = v_data[k];
        x, y = simple_training_data(x_template, y_template, vx_data, vy_data)
        push!(X_data, x);
        push!(Y_data, y);
    end
    println("Completed reading $raw_data_file")
end
println("Completed creating Training Data")

if isdir(training_datafolder) println("$training_datafolder already exists");
else mkdir(training_datafolder); end

f = h5open("$training_datafolder/$training_datafile.h5","w");
for i in 1:length(X_data)
    write(f, "X_data$i", X_data[i], "Y_data$i", Y_data[i]);
end
close(f);


#-----------------------------------------------------------------------------------------------------------------------------------------
# Building an initial MPO W
#-----------------------------------------------------------------------------------------------------------------------------------------
L = length(X_data[1]);
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
W_init = MPO(L);
for i in 1:L W_init[i] = randomITensor(inds(X_data[1][i], "x")[1], inds(Y_data[1][i], "y"), ind_b[i], ind_b[i+1]); end # W is a MPO now
#W_init = (1/sqrt(inner(W_init,W_init)))*W_init;
#W = orthogonalize(W, 1);
#------------------------------------------------------------------------------------------------------------------------------------------

W, err = find_optimal_W_PRE(X_data, Y_data, D, alpha, tolerance, max_steps2, W_init);
cycles = 1;
while err>tolerance && cycles<=ncycles 
    println("Cycle Number ", cycles);
    global err;
    global W;
    global cycles;
    W, err = find_optimal_W(X_data, Y_data, D, alpha, eta, tolerance, max_steps1, W);
    W, err = find_optimal_W_PRE(X_data, Y_data, D, alpha, tolerance, max_steps2, W);
    cycles += 1;
end

if isdir("Optimal_W_MPOs") println("Optimal_W_MPOs folder already exists");
else mkdir("Optimal_W_MPOs"); end

f = h5open("Optimal_W_MPOs/$Optimal_MPO_name.L$L.D$D.alpha$alpha.h5","w")
write(f,"Optimal_W", W, "Error", err)
close(f)