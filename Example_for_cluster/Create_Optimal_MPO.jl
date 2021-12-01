using ITensors
using ITensors.HDF5
include("MPS_functions.jl")

training_datafolder = "Training_Data";
D = 10;
alpha = 0.01;
eta = 0.02;
tolerance = 0.01;
max_steps = 100;

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