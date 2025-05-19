# Plan

## Tools

- VSCode
- ssh
- Python
- Conda
- Jupyter Notebooks
- JAX with GPU
- AI use
- GitHub

## Schedule
- ssh to the server
- create conda env `conda create -n demo python` and `conda activate demo`
- install JAX and other dependencies `pip install jax[cuda12] numpy tqdm matplotlib`
- ask gpt o3 to write a Vlasov PIC solver with JAX
- benchmark the numpy version
- benchmark the jax version on the CPU, without JIT
- benchmark the jax version on the CPU, with JIT
- benchmark the jax version on the GPU (100x speedup)
- add collisions
- walk through codebase https://github.com/Vilin97/Vlasov-Landau-SBTM

## Discussion
- Do you write tests?
- Do you do interactive plotting? What library?
- 