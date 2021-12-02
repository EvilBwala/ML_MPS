"""
This code creates generic random raw data and stores them in a folder
INPUT:
ARGS[1] = L is the length of the input and output sequences
ARGS[2] = sig_dims is the dimensionality of the spins e.g. 2 for Michael's system
ARGS[3] = tau_dims is the dimensionality of the output feature space e.g. 2 for our case
ARGS[4] = num_signals is the number of specific protocols (signals) we want to train our system against
ARGS[5] = num_trials is the number of trials for a spicific protocol (signal)
ARGS[6] = Raw_data_folder_name is the name of the folder where Raw data is going to be stored
"""

using Distributions
using NPZ
using LinearAlgebra
using Random

L = parse(Int, ARGS[1]);
sig_dims = parse(Int, ARGS[2]);
tau_dims = parse(Int, ARGS[3]);
num_signals = parse(Int, ARGS[4]);
num_trials = parse(Int, ARGS[5]);
Raw_data_folder_name = ARGS[6];

if isdir(Raw_data_folder_name)
    println("$Raw_data_folder_name folder already exists");
else
    mkdir(Raw_data_folder_name);
end

for i in 1:num_signals
    b = Binomial();
    v_data = [rand(b, L) .+ 1 for i in 1:num_trials];
    v_arr = zeros(num_trials, L);
    for j in 1:num_trials
        v_arr[j, :] = v_data[j];
    end
    ys = collect(1:L);
    ys = cos.(pi*ys/(10*rand(1)));
    nr = Normal();
    ns = rand(nr, L);
    ys = ys .+ 0.01 .* ns;

    npzwrite("$Raw_data_folder_name/Raw_Data$i", Dict("Pattern_list" => v_arr, "Protocol" => ys));
    println("Random Data $i is created");
end
