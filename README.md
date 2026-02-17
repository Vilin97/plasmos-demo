## Tools to demo:
- GitHub (version control)
- Weights & Biases (experiment logging, reports)
- Claude Code (interactive agent mode)
- Hyak (interactive and batch jobs)
- multi-gpu setup

## Examples to use:
- PIC for collisionless vlasov equation
  - showcase: W&B, CC, Hyak, multi-gpu
  - from scratch, in jax, first on a single gpu, then on multi-gpu
- PIC for collisional vlasov-landau equation
  - showcase: W&B, GitHub
  - already implemented, and experiments logged -- show W&B logging, reports and data loading

## Schedule
- create conda env `conda create -n plasmos-demo python` and `conda activate plasmos-demo`, install JAX, W&B and other dependencies `pip install jax[cuda12] numpy tqdm matplotlib wandb`
- start with vlasov.py -- a barebones implementation of a PIC solver for the Vlasov equation
- add multi-gpu support (use all available gpus, and all available memory). Should still be runnable on a cpu if no gpu is available.
- make a runnable script with a `main()` that can accept: 
  - dimension dv
  - number of particles n
  - number of cells M
  - time step dt
- add wandb logging: args used, host device(s), peak memory, runtime, norm of Electric energy, fitted slope, the electric energy plot. Log to `entity='naske'`.
- make a batch job, with dv=1, n=10^6, M=100, dt=0.01 by default
  - sweep n in 1e6, 1e7, 1e8
  - sweep M in 20, 100, 400
  - sweep dt in 1e-1, 1e-2, 1e-3
  - sweep dv in 1,3
  - sweep seed in 1,2,3
- make a report in W&B: [link](https://wandb.ai/naske/vlasov-pic/reports/Untitled-Report--VmlldzoxNTk2MTMyOQ?accessToken=qvujtaphnbcsc2z1i89eiexugwmua1qnc8flmkazctwvx0sp4130f7hcf0466m71)
