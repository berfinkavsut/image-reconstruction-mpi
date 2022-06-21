## Comparison of ADMM and Kaczmarz REconstruction Algorithms 

Project members: Aslı Alpman, Beril Alyüz, Berfin Kavşut

System Function Reconstruction for MPI, Magnetic Particle Imaging, leads to an inverse problem which is
mostly ill-conditioned. In this project, we implemented Kaczmarz and ADMM, Alternating Direction
Method of Multipliers, algorithms to solve this inverse problem. With ADMM, we also employed total
variation and l norm regularization to utilize slowly changing nature and the sparsity of the images used 1
in the project. We compared the performance of Kaczmarz and ADMM by looking at the quality of the
resulted images and convergence properties.

# Sources
Datasets are taken from [OpenMPI datasets]([url](https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/index.html))
Source code: https://web.stanford.edu/~boyd/papers/admm/
