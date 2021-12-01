First run the Create_Input_Output_templates.jl file to generate appropriate template

julia Create_Input_Output_templates.jl $L $sig_dims $tau_dims


Next run the Create_training_data.jl file to create appropriate training data. This step should be run on the cluster as every raw data file is going to provide a separate training data. So everything can be run in parallel in this step.

julia Create_training_data.jl $L $sig_dims $tau_dims $D $raw_datfile $eta $max_steps $tolerance


Finally run the Create_Optimal_MPO.jl file

julia Create_Optimal_MPO.jl $training_datafolder $D $alpha $eta $tolerance $max_steps


Please read the documentation of these three Create* files to understand the variables sig_dims, tau_dims, D etc.

The raw datafile needs to be in npz format with dictionary indices, "Pattern_list" and "Protocol"
Pattern_list would correspond to the pattern by num_trials array (i.e. Each row corresponding to a pattern, so the shape of the matrix would be num_trials by pattern_length). Also the values needs to be positive integers. For a up-down spin system, the patterns have to be [1 2 1 1 2 1 .... 1 2] instead of [1 -1 1 1 -1 1 .... 1 -1]. imilarly for spins with dimensionality =3, the patterns can be [1 1 2 3 1 3 1 ....2 3 1 1].
Protocol would correspond to the array containing protocol values at the last n (=20) sites.
