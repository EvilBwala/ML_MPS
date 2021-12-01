First run the Create_Input_Output_templates.jl file to generate appropriate template

julia Create_Input_Output_templates.jl $L $sig_dims $tau_dims


Next run the Create_training_data.jl file to create appropriate training data. This step should be run on the cluster as every raw data file is going to provide a separate training data. So everything can be run in parallel in this step.

julia Create_training_data.jl $L $sig_dims $tau_dims $D $raw_datfile $eta $max_steps $tolerance


Finally run the Create_Optimal_MPO.jl file

julia Create_Optimal_MPO.jl $training_datafolder $D $alpha $eta $tolerance $max_steps


Please read the documentation of these three Create* files to understand the variables sig_dims, tau_dims, D etc.
