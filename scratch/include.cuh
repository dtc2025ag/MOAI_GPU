#ifndef INCLUDE_H
#define INCLUDE_H

#pragma once

//intel hexl library 
// #include "hexl/hexl.hpp"

//SEAL library
// #include "seal/seal.h"

// PhantomFHE
#include "phantom.h"

// evaluator
// #include "ckks_evaluator.cuh"
#include "ckks_evaluator_parallel.cuh"

//C++
#include <iostream>
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <functional>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <string>
#include <memory>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <iomanip>

#include <chrono>



//source code
#include "utils.cuh"
#include "utils_moai.cuh"
#include "Ct_pt_matrix_mul.cuh"
#include "test_phantom_ckks.cuh"
#include "Batch_encode_encrypt.cuh"
#include "Ct_ct_matrix_mul.cuh"
#include "gelu_other.cuh"
#include "layernorm.cuh"
#include "softmax.cuh"
#include "single_att_block.cuh"




#endif


