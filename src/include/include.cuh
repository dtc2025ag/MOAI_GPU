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
#include "source/ckks_evaluator_parallel.cuh"

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
#include "source/bootstrapping/Bootstrapper.cuh"
#include "source/utils.cuh"
#include "source/utils_moai.cuh"
#include "source/matrix_mul/Batch_encode_encrypt.cuh"
#include "source/matrix_mul/Ct_pt_matrix_mul.cuh"
#include "source/matrix_mul/Ct_ct_matrix_mul.cuh"
#include "source/non_linear_func/gelu_other.cuh"
#include "source/non_linear_func/layernorm.cuh"
#include "source/non_linear_func/softmax.cuh"
#include "source/att_block/single_att_block.cuh"


//test code
#include "test/test_phantom_ckks.cuh"
#include "test/matrix_mul/test_batch_encode_encrypt.cuh"
#include "test/matrix_mul/test_ct_pt_matrix_mul.cuh"
#include "test/matrix_mul/test_ct_ct_matrix_mul.cuh"
#include "test/non_linear_func/test_gelu.cuh"
#include "test/non_linear_func/test_layernorm.cuh"
#include "test/non_linear_func/test_softmax.cuh"
#include "test/test_single_layer.cuh"

#endif


