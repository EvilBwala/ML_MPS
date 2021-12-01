
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
ARGS[6] = max_steps is the total number ofssteps we want our algorithm to run
OUTPUT:
The optimal W which is created in stored in a folder named Optimal_W_MPOs under the name
Optimal_W.h5.
"""

using ITensors
using ITensors.HDF5
include("MPS_functions.jl")

training_datafolder = ARGS[1];
D = parse(Int, ARGS[2]);
alpha = parse(Float64, ARGS[3]);
eta = parse(Float64, ARGS[4]);
tolerance = parse(Float64, ARGS[5]);
max_steps = parse(Int, ARGS[6]);

t_data_files = filter(z->occursin("h5", z) , readdir(training_datafolder));

N_data = length(t_data_files);
X_data = Vector{MPS}();
Y_data = Vector{MPS}();

for fl in t_data_files
    f = h5open("$training_datafolder/"*fl, "r");
    psi = read(f, "psi", MPS);
    y = read(f, "y", MPS);
    push!(X_data, psi);
    push!(Y_data, y);
    close(f);
end

W = find_optimal_W(X_data, Y_data, D, alpha, eta, tolerance, max_steps);

if isdir("Optimal_W_MPOs")
    println("Optimal_W_MPOs folder already exists");
else
    mkdir("Optimal_W_MPOs");
end

f = h5open("Optimal_W_MPOs/Optimal_W.h5","w")
write(f,"Optimal_W",W)
close(f)