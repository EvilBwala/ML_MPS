#!/bin/bash

# Filename information
sourcefile=Create_Optimal_MPO.jl    # Name of the julia file that creates the Optimal MPO
sourcedirectory=ML_MPS              # Name of the folder which contains teh important files
filename=Testing                    # Name of directory where the Raw_datafolder exists. Place it at the same level at the MP_MPS folder


# Fixed simulation parameters
training_datafolder="Training_data"     # Name of the folder where all the Training data is stored
eta=0.01                                # Step-size for gradient descent
tolerance=0.01                          # Tolerance of error
max_steps1=5                            # Steps for running find_optimal_W.jl
max_steps2=5                            # Steps for running find_optimal_W_PRE.jl
ncycles=2                               # Number of alternating cycles of the two algorithms

# Variable parameters
Ds=(10)                                 # Maximum Bond Dimensions of the MPO W
alphas=(0.0001)                         # The regularizer

cd ..

if [ ! -d "$filename" ]
then
    mkdir $filename
fi

cp ${sourcedirectory}/* ${filename}/
cd $filename


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
        echo "julia ${sourcefile} $training_datafolder $D $alpha $eta $tolerance $max_steps1 $max_steps2 $ncycles" >> submit.sbatch
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        # For running on cluster : uncomment the next line and comment the 4th line from here 
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        #qsub submit.sbatch
        #module load julia
        #julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "TensorOperations"])'
        julia ${sourcefile} $training_datafolder $D $alpha $eta $tolerance $max_steps1 $max_steps2 $ncycles
    done
done

cd ..
cd ${sourcedirectory}/

