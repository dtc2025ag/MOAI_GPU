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

#include <cstdint>
static inline std::size_t ciphertext_device_bytes(const PhantomCiphertext &ct)
{
    if (ct.data() == nullptr)
        return 0;
    return ct.size() * ct.coeff_modulus_size() * ct.poly_modulus_degree() * sizeof(uint64_t);
}

static inline std::size_t output_device_bytes(const std::vector<PhantomCiphertext> &output)
{
    std::size_t total = 0;
    for (const auto &ct : output)
        total += ciphertext_device_bytes(ct);
    return total;
}

// 打印为 MB
static inline void print_output_mem(const std::vector<PhantomCiphertext> &output)
{
    const double MB = 1024.0 * 1024.0;
    std::size_t total = output_device_bytes(output);
    std::cout << "[GPU] output device memory ~ " << (total / MB) << " MB" << std::endl;
}

static inline void print_cuda_meminfo(const char *tag)
{
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const double MB = 1024.0 * 1024.0;
    std::cout << tag << " Free: " << free_b / MB << " MB, Total: " << total_b / MB << " MB\n";
}

using phantom::util::cuda_stream_wrapper;
using std::vector;
vector<cuda_stream_wrapper> stream_pool; 


static inline void bridge_to_default(const cuda_stream_wrapper &sw)
{
    auto dst = phantom::util::global_variables::default_stream->get_stream();
    if (sw.get_stream() == dst)
        return; 

    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, sw.get_stream()); 
    cudaStreamWaitEvent(dst, ev, 0);      
    cudaEventDestroy(ev);                 
}

PhantomCiphertext deep_copy_cipher(const PhantomCiphertext &src,
                                   const PhantomContext &context,
                                   phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
    PhantomCiphertext dst;

    
    dst.set_chain_index(src.chain_index());
    dst.set_poly_modulus_degree(src.poly_modulus_degree());
    dst.set_coeff_modulus_size(src.coeff_modulus_size());
    dst.set_scale(src.scale());
    dst.set_correction_factor(src.correction_factor());
    dst.set_ntt_form(src.is_ntt_form());
    dst.SetNoiseScaleDeg(src.GetNoiseScaleDeg());

    
    dst.reinit_like(src.size(),
                    src.coeff_modulus_size(),
                    src.poly_modulus_degree(),
                    stream.get_stream());

    
    size_t count = src.size() * src.coeff_modulus_size() * src.poly_modulus_degree();
    cudaMemcpyAsync(dst.data(), src.data(),
                    count * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice,
                    stream.get_stream());

    return dst;
}

vector<PhantomCiphertext> deep_copy_cipher(const vector<PhantomCiphertext> &src,
                                           const PhantomContext &context,
                                           phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
    vector<PhantomCiphertext> dst(src.size());

    for (size_t i = 0; i < src.size(); i++)
    {
        
        dst[i].set_chain_index(src[i].chain_index());
        dst[i].set_poly_modulus_degree(src[i].poly_modulus_degree());
        dst[i].set_coeff_modulus_size(src[i].coeff_modulus_size());
        dst[i].set_scale(src[i].scale());
        dst[i].set_correction_factor(src[i].correction_factor());
        dst[i].set_ntt_form(src[i].is_ntt_form());
        dst[i].SetNoiseScaleDeg(src[i].GetNoiseScaleDeg());

        
        dst[i].reinit_like(src[i].size(),
                           src[i].coeff_modulus_size(),
                           src[i].poly_modulus_degree(),
                           stream.get_stream());

        
        size_t count = src[i].size() * src[i].coeff_modulus_size() * src[i].poly_modulus_degree();
        cudaMemcpyAsync(dst[i].data(), src[i].data(),
                        count * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice,
                        stream.get_stream());
    }

    return dst;
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
#include "source/matrix_mul/Rotary_Position_Embedding.cuh"

// test code
#include "test/test_phantom_ckks.cuh"
#include "test/matrix_mul/test_batch_encode_encrypt.cuh"
#include "test/matrix_mul/test_ct_pt_matrix_mul.cuh"
#include "test/matrix_mul/test_ct_ct_matrix_mul.cuh"
#include "test/non_linear_func/test_gelu.cuh"
#include "test/non_linear_func/test_layernorm.cuh"
#include "test/bootstrapping/bootstrapping.cuh"
#include "test/non_linear_func/test_softmax.cuh"
#include "test/non_linear_func/test_BPmax_BatchLN.cuh"
#include "test/test_single_layer.cuh"
#include "test/matrix_mul/test_rotary_position_embedding.cuh"

#endif
