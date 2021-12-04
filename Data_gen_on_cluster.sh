#!/bin/bash

# Filename information
sourcefile=Create_training_data.jl  # Name of the julia file which creates training data from raw data
sourcedirectory=ML_MPS              # Name of the folder which contains all the important files
filename=Testing                    # Name of directory where the Raw_datafolder exists

# Fixed simulation parameters
L=10                                # The length of the pattern sequence
sig_dims=2                          # Spin dimensions = 2 for Michael's system
tau_dims=2                          # Feature space physical dimension = 2 for our case
D=4                                 # Maximum bond dimension of psi MPS
raw_datfolder=Raw_data_example      # Name of folder which contains Raw data
eta=0.01                            # Step size of gradient descent
max_steps=50                        # Max steps for running algo
tolerance=0.01                      # Tolerance of error
training_datafolder=Training_data   # Name of the folder which will contain the training data built from raw data 


cd ..

if [ ! -d "$filename" ]
then
    mkdir $filename
fi
cp ${sourcedirectory}/* ${filename}/
cd $filename


# Variable parameters
raw_datfiles=$(ls $raw_datfolder)

#---------------------------------------------------------------------------------------------
# Uncomment the next line if the packages NPZ, Distributions and ITensors are not installed
#julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "ITensors"])'
#---------------------------------------------------------------------------------------------
# Create the input and output templates
julia Create_Input_Output_templates.jl $L $sig_dims $tau_dims 


if [ ! -d "$training_datafolder" ]
then
    mkdir $training_datafolder
fi
#cp * ${datafolder}/
#cd $datafolder


for raw_datfile in ${raw_datfiles[@]}
do
    echo "#!/bin/sh" > submit.sbatch
    echo "#SBATCH --job-name=CHF.$L.$raw_datfile" >> submit.sbatch
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
    echo "julia ${sourcefile} $L $sig_dims $tau_dims $D $raw_datfolder $raw_datfile $eta $max_steps $tolerance $training_datafolder" >> submit.sbatch
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # For running on cluster : uncomment the next line and comment the 4th line from here 
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    #qsub submit.sbatch
    #module load julia
    #julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "TensorOperations"])'
    julia ${sourcefile} $L $sig_dims $tau_dims $D $raw_datfolder $raw_datfile $eta $max_steps $tolerance $training_datafolder
done

cd ..
cd ${sourcedirectory}/

