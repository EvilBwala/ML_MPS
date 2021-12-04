# ML_MPS
ML using MPS

Step 1:  
Run the Create_Input_Output_templates.jl file to generate appropriate template  
julia Create_Input_Output_templates.jl $L $sig_dims $tau_dims  


Step 2:  
Run the Gen_random_raw_data.jl file to create random raw data  
julia Gen_random_raw_data.jl $L $sig_dims $tau_dims $num_signals $num_trials $Raw_data_folder_name  
If you already have raw data, then you can skip step 2.  


Step 3:  
Run the Create_training_data.jl file to create appropriate training data. This step should be run on the cluster as every raw data file is going to provide a separate training data. So everything can be run in parallel in this step.  
julia Create_training_data.jl $L $sig_dims $tau_dims $D $raw_datafolder $raw_datfile $eta $max_steps $tolerance $training_datafolder  

To run Step 3 on cluster, run the Data_gen_on_cluster.sh file with appropriate parameters  

(NOTE 1: As it is now, the feature space for the output as is coded in the Create_training_data.jl file corresponds to $tau_dims = 2. For different $tau_dims, make appropriate changes in this file to variable "vy_feature".)  

(NOTE 2 : $raw_datafolder and $Raw_data_folder_name should be the same)  

Step 4:  
Run the Create_Optimal_MPO.jl file in the working directory  
julia Create_Optimal_MPO.jl $training_datafolder $D $alpha $eta $tolerance $max_steps1 $max_steps2 $ncycles  
Otherwise you can also run the Create_MPO.sh file with appropriate parameters  

Please read the documentation of these three Create* and *sh files to understand the variables sig_dims, tau_dims, D etc.  

The raw datafile needs to be in npz format with dictionary indices, "Pattern_list" and "Protocol"  
Pattern_list would correspond to the pattern by num_trials array (i.e. Each row corresponding to a pattern, so the shape of the matrix would be num_trials by pattern_length). Also the values needs to be positive integers. For a up-down spin system, the patterns have to be [1 2 1 1 2 1 .... 1 2] instead of [1 -1 1 1 -1 1 .... 1 -1]. Similarly for spins with dimensionality =3, the patterns can be [1 1 2 3 1 3 1 ....2 3 1 1].  
Protocol would correspond to the array containing protocol values at the last n (=20) sites.  
