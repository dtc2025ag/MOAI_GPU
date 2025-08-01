#ifndef INCLUDE_H
#define INCLUDE_H

#pragma once

//intel hexl library 
// #include "hexl/hexl.hpp"

//SEAL library
#include "seal/seal.h"

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

#include "source/matrix_mul/Batch_encode_encrypt.hpp"
#include "source/matrix_mul/Ct_pt_matrix_mul.hpp"
#include "source/matrix_mul/Ct_ct_matrix_mul.hpp"

#include "source/non_linear_func/softmax.hpp"
#include "source/non_linear_func/layernorm.hpp"
#include "source/non_linear_func/gelu.hpp"
#include "source/non_linear_func/gelu_others.hpp"

#include "source/att_block/single_att_block.hpp"

//test
// #include "test/test_pt_att.hpp"
#include "test/test_SEAL_ckks.hpp"

#include "test/matrix_mul/test_batch_encode_encrypt.hpp"
#include "test/matrix_mul/test_ct_pt_matrix_mul.hpp"
#include "test/matrix_mul/test_ct_ct_matrix_mul.hpp"

//Please choose one of the following 4 items. 

//#include "test/att_block/test_12_att_block.hpp"
//#include "test/bootstrapping/test_layernorm_bootstrapping.hpp"
//#include "test/test_single_layer.hpp"
#include "test/test_full_scheme.hpp"





#endif
