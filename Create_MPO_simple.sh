#!/bin/bash

# Filename information
sourcefile=Create_Optimal_MPO_simple.jl    # Name of the julia file that creates the Optimal MPO
sourcedirectory=ML_MPS              # Name of the folder which contains teh important files
filename=Testing_ML                    # Name of directory where the Raw_datafolder exists. Place it at the same level at the ML_MPS folder


# Fixed simulation parameters
raw_datafolder="Simple_Raw_Data"     # Name of the folder where all the Training data is stored
L=10
sig_dims=3
tau_dims=2
num_trials=4
training_datafolder="Simple_TD"
training_datafile="All_Data"
Optimal_MPO_name="goodMPO"
eta=0.01                                # Step-size for gradient descent
tolerance=0.01                          # Tolerance of error
max_steps1=10                            # Steps for running find_optimal_W.jl
max_steps2=10                            # Steps for running find_optimal_W_PRE.jl
ncycles=10                               # Number of alternating cycles of the two algorithms

# Variable parameters
Ds=(10 20) #40 80 160)                                 # Maximum Bond Dimensions of the MPO W
alphas=(0.00000001)                         # The regularizer

cd ..

if [ ! -d "$filename" ]
then
    mkdir $filename
fi

cp ${sourcedirectory}/* ${filename}/
cp -r ${raw_datafolder} ${filename}/
cd $filename


#---------------------------------------------------------------------------------------------
# Uncomment the next line if the packages NPZ, Distributions and ITensors are not installed
#julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "ITensors"])'
#---------------------------------------------------------------------------------------------
# Create the input and output templates
julia Create_Input_Output_templates.jl $L $sig_dims $tau_dims 


for D in ${Ds[@]}
do
    for alpha in ${alphas[@]}
    do
        echo "#!/bin/sh" > submit.sbatch
        echo "#SBATCH --job-name=CHF.$D.$alpha" >> submit.sbatch
        echo "#SBATCH --output=./%j.out" >> submit.sbatch
        echo "#SBATCH --error=./%j.err" >> submit.sbatch
        echo "#SBATCH --partition=broadwl" >> submit.sbatch
        echo "#SBATCH --account=pi-svaikunt" >> submit.sbatch
        echo "#SBATCH --constraint=ib" >> submit.sbatch
        echo "#SBATCH --nodes=1" >> submit.sbatch
        echo "#SBATCH --ntasks-per-core=1" >> submit.sbatch
        echo "#SBATCH --time=15:00:00" >> submit.sbatch
        #echo "#SBATCH --mail-type=ALL" >> submit.sbatch
        #echo "#SBATCH --mail-user=agnish@uchicago.edu" >> submit.sbatch
        echo "#SBATCH --mem-per-cpu=8000" >> submit.sbatch
        echo "module load julia" >> submit.sbatch
        echo "julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "ITensors"])'" >> submit.sbatch
        echo "time julia ${sourcefile} $raw_datafolder $L $sig_dims $tau_dims $num_trials $training_datafolder $training_datafile $D $alpha $eta $tolerance $max_steps1 $max_steps2 $ncycles $Optimal_MPO_name" >> submit.sbatch
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        # For running on cluster : uncomment the next line and comment the 4th line from here 
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        #qsub submit.sbatch
        #module load julia
        #julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "TensorOperations"])'
        time julia ${sourcefile} $raw_datafolder $L $sig_dims $tau_dims $num_trials $training_datafolder $training_datafile $D $alpha $eta $tolerance $max_steps1 $max_steps2 $ncycles $Optimal_MPO_name
    done
done

cd ..
cd ${sourcedirectory}/
