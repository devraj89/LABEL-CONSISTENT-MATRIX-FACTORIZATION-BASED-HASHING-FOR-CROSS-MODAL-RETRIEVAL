This is the implementation for the paper 
"Label Consistent Matrix Factorization based Hashing for Cross Modal Retrieval"
ICIP 2017.
The poster can be found here : 
https://sigport.org/documents/label-consistent-matrix-factorization-based-hashing-cross-modal-retrieval

The main paper can be found here 
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8296813

********************************************************************************
Please download the datasets as instructed in the paper and put it in a folder titled
datasets\ . For help regarding the datasets kindly please drop me a mail.
********************************************************************************


********************************************************************************
The basic essence of my algorithm can be found in the two files 

[1] solveUCMFH_devraj3.m -- This is the ours1 implementation of our paper 
Kindly please select the parameters appropriately to get the best possible results.

[2] solveUCMFH_devraj3_propagate -- This is the proposed techniue to handle large 
amounts of data. Kindly please the implementation in the mirflickr and nus-wide dataset
codes. Also select the value of rho appropriately.

********************************************************************************

********************************************************************************
I am providing the details of the operation for the three datasets 

(1) Wikipedia dataset -- please run wiki_ours1.m & wiki_ours2.m
The ours2 version does not re-generate the hash codes of the retrieval set!
so results are much better.

(2) MirFlickr dataset -- please run the program mirflickr_ours2.m to get a sense of
how to use the ours2 version of the algorithm. In this we initialize the first 
batch of the data using solveUCMFH_devraj3 and then propagate the learned variables 
through the code solveUCMFH_devraj3_propagate

Kindly please select the value of rho appropriately. Also you need to select the subset
size of the data appropriately. The larger the subset size the better is the overall performance
though at the cost of computational power.

I have not provided the code for the ours1 version for the MirFlickr dataset. It is quite
easy to write it. Just take the number of samples N=5000 and use the solveUCMFH_devraj3.m to learn
the hash codes as shown for the Wikipedia dataset.

(3) Nuswide dataset -- The same as the MirFlickr dataset. Kindly please follow the same instructions!

********************************************************************************






********************************************************************************
I am also providing here some extra stuff: Please look into the code 
wiki_extra_with_projections_and_stuff.m to understand the implementations

(1) Suppose you make the latent factors V common -- use this codes 
solveUCMFH_devraj6_proj -- the basic implementation
solveUCMFH_devraj6_proj_propagate -- the implementation to handle large amounts of data

(2) Suppose you even make the U's common -- use this codes
solveUCMFH_devraj7_proj -- the basic implementation
use the following options 
option = 1 -- use pca projections 
option = 2 -- use cca projections 
option = 6 -- use random initializations/projections

solveUCMFH_devraj7_proj_propagate -- the implementation to handle large amounts of data

This extra implementations are not used in the ICIP paper but have been observed
to give even better results as compared to what is reported.
********************************************************************************

