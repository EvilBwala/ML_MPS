"""
This code creates appropriate training data from raw data files and stores them in a folder named Training_Data
INPUT:
ARGS[1] = L is the length of patterns
ARGS[2] = sig_dims is the dimensionsionality of the spins e.g  2 for Michael's system
ARGS[3] = tau_dims is the dimensionality of the feature space output which is 2 for our case
ARGS[4] = D is the maximum bond dimension of the psi input to be created
ARGS[5] = raw_datfolder is the name of the folder containing raw_datafiles
ARGS[6] = raw_datfile is the name of the file containing raw data
            It is supposed to be a npz file containing "Pattern_list" and "Protocol"
            as dictionary indices for the Patetrn data and the protocol at the last n sites.
ARGS[7] = eta is the step size of the gradient descent for finding optimal psi
ARGS[8] = max_steps is the maximum steps for which the optimal psi algo is run
ARGS[9] = tolerance is the error tolerance for building psi
ARGS[10] = training_datafolder is the name of the folder where training data is to be stored
OUTPUT:
The codes builds psi and the feature space output y
These are stored in a HDF5 file in the folder Training_Data
Psi is stored at distionary index psi and y is tored at dictionary index y
"""

using ITensors
using ITensors.HDF5
using Distributions
using NPZ
include("Data_gen_PRX.jl")

L = parse(Int, ARGS[1]) # Length of the patterns
sig_dims = parse(Int, ARGS[2]) # Dimensions of the spins = 2 for Michael's system
tau_dims = parse(Int, ARGS[3]) # Dimensions of the output = 2 (sinh(y) and cosh(y) in our case)
D = parse(Int, ARGS[4]) # Maximum Bond Dimension of the input (psi)
raw_datfolder = ARGS[5] # Name of folder where raw data files are stored
raw_datfile = ARGS[6] # Name of the file containing raw data
eta = parse(Float64, ARGS[7]); # Step size during optimizing the input data into psi
max_steps = parse(Int, ARGS[8]); # Maximum number of steps
tolerance = parse(Float64, ARGS[9]); # Tolerance during building the input psi
training_datafolder = ARGS[10] # Name of folder to store Training data

f = h5open("IO_Templates/xy_template.L$L.sig$sig_dims.tau$tau_dims.h5","r");
x_template = read(f, "x_template", MPS);
y_template = read(f, "y_template", MPS);
close(f);


if isdir(training_datafolder) println("$training_datafolder already exists");
else mkdir(training_datafolder); end

#--------------------------------------------------------------------------------------------------------------------
raw_data = npzread("$raw_datfolder/$raw_datfile");
v_arr = trunc.(Int, raw_data["Pattern_list"]);
v_data = [v_arr[i,:] for i in 1:size(v_arr)[1]];

for k in 1:size(v_arr)[1]
    println("Trial ", k);
    vy_data = raw_data["Protocol"]
    vx_data = v_data[k];
    x, y = simple_training_data(x_template, y_template, vx_data, vy_data);
    f1 = h5open("$training_datafolder/$raw_datfile.L$L.sig$sig_dims.tau$tau_dims.trial$k.h5","w")
    write(f1,"x",x, "y", y);
    close(f1);
end


