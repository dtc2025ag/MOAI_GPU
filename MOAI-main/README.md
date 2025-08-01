# MOAI

# 1. Install on Ubuntu/Debian
## Install dependent libraries
```
sudo apt update
sudo apt install cmake g++ git libntl-dev libssl-dev libgmp-dev pkg-config
```

## Install the third party SEAL globally
```
cd thirdparty/SEAL-4.1-bs
// if build exist, run
// rm -rf build
cmake -S . -B build
cmake --build build
sudo cmake --install build
```

# 2. Go to main folder and Run
```
cmake -S . -B build
cd build
make
./test
```

# 3. Test result
```
All time cost results outputted is the total time of 256 inputs (each input has up to 128 tokens).
Please divide by 256 to get the amortized time. 
