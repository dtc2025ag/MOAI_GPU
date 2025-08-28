#ifndef INCLUDE_H
#define INCLUDE_H

#pragma once

// intel hexl library
//  #include "hexl/hexl.hpp"

// SEAL library
//  #include "seal/seal.h"

// PhantomFHE
#include "phantom.h"

// evaluator
// #include "ckks_evaluator.cuh"
#include "source/ckks_evaluator_parallel.cuh"

// C++
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

using phantom::util::cuda_stream_wrapper;
using std::vector;
vector<cuda_stream_wrapper> stream_pool; // 线程私有流池

// 把“自定义流上的完成”桥接到默认流，避免默认流读到半成品
static inline void bridge_to_default(const cuda_stream_wrapper &sw)
{
    auto dst = phantom::util::global_variables::default_stream->get_stream();
    if (sw.get_stream() == dst)
        return; // 同一条流就不需要桥接

    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, sw.get_stream()); // 在生产流上记录事件
    cudaStreamWaitEvent(dst, ev, 0);      // 让“库的默认流”等待事件
    cudaEventDestroy(ev);                 // 等待已入队，销毁事件对象即可
}

// source code
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

// test code
#include "test/test_phantom_ckks.cuh"
#include "test/matrix_mul/test_batch_encode_encrypt.cuh"
#include "test/matrix_mul/test_ct_pt_matrix_mul.cuh"
#include "test/matrix_mul/test_ct_ct_matrix_mul.cuh"
#include "test/non_linear_func/test_gelu.cuh"
#include "test/non_linear_func/test_layernorm.cuh"
#include "test/non_linear_func/test_softmax.cuh"
#include "test/test_single_layer.cuh"

#endif
