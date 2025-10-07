# MOAI_GPU
The code has been tested on Nvidia H200, A100.
## 1. Reuqirements
```
CUDA version=11.8
cmake version>=3.20
gcc,g++ version=11.4.0
OpenMP version=4.5
```

## 2. Go to main folder and Run
```
cmake -S . -B build
cd build
make
# Optimal omp thread number varies from different CPU and GPU combination, empirically 4-8.
OMP_NUM_THREADS=4 ./test
```

## 3. Test result
```
All time cost results outputted is the total time of 256 inputs (each input has up to 128 tokens).
Please divide by 256 to get the amortized time. 
```

## Citation
```
@misc{cryptoeprint:2025/991,
      author = {Linru Zhang and Xiangning Wang and Jun Jie Sim and Zhicong Huang and Jiahao Zhong and Huaxiong Wang and Pu Duan and Kwok Yan Lam},
      title = {{MOAI}: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/991},
      year = {2025},
      url = {https://eprint.iacr.org/2025/991}
}
```
