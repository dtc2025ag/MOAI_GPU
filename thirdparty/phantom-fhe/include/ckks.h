#pragma once

#include "context.cuh"
#include "fft.h"
#include "ntt.cuh"
#include "plaintext.h"
#include "rns.cuh"

#include <complex>
#include <cuComplex.h>

class PhantomCKKSEncoder {

private:

    uint32_t slots_{};
    uint32_t sparse_slots_ = 0;
    uint32_t decoding_sparse_slots_ = 0;
    std::unique_ptr<phantom::util::ComplexRoots> complex_roots_;
    std::vector<cuDoubleComplex> root_powers_;
    std::vector<uint32_t> rotation_group_;
    std::unique_ptr<DCKKSEncoderInfo> gpu_ckks_msg_vec_;
    uint32_t first_chain_index_ = 1;

    void encode_internal(const PhantomContext &context,
                         const cuDoubleComplex *values, size_t values_size,
                         size_t chain_index, double scale,
                         PhantomPlaintext &destination,
                         const cudaStream_t &stream);

    inline void encode_internal(const PhantomContext &context,
                                const double *values, size_t values_size,
                                size_t chain_index, double scale,
                                PhantomPlaintext &destination,
                                const cudaStream_t &stream) {
        std::vector<cuDoubleComplex> input(values_size);
        for (size_t i = 0; i < values_size; i++) {
            input[i] = make_cuDoubleComplex(values[i], 0.0);
        }
        encode_internal(context, input.data(), values_size, chain_index, scale, destination, stream);
    }

    inline void encode_internal(const PhantomContext &context,
                                const std::complex<double> *values, size_t values_size,
                                size_t chain_index, double scale,
                                PhantomPlaintext &destination,
                                const cudaStream_t &stream) {
        std::vector<cuDoubleComplex> input(values_size);
        for (size_t i = 0; i < values_size; i++) {
            input[i] = make_cuDoubleComplex(values[i].real(), values[i].imag());
        }
        encode_internal(context, input.data(), values_size, chain_index, scale, destination, stream);
    }

    void decode_internal(const PhantomContext &context,
                         const PhantomPlaintext &plain,
                         cuDoubleComplex *destination,
                         const cudaStream_t &stream);

    inline void decode_internal(const PhantomContext &context,
                                const PhantomPlaintext &plain,
                                double *destination,
                                const cudaStream_t &stream) {
        auto decoding_sparse_slots = decoding_sparse_slots_ == 0 ? sparse_slots_ : decoding_sparse_slots_;
        std::vector<cuDoubleComplex> output(decoding_sparse_slots);
        decode_internal(context, plain, output.data(), stream);
        for (size_t i = 0; i < decoding_sparse_slots; i++)
            destination[i] = output[i].x;
    }

public:

    explicit PhantomCKKSEncoder(const PhantomContext &context);

    PhantomCKKSEncoder(const PhantomCKKSEncoder &copy) = delete;

    PhantomCKKSEncoder(PhantomCKKSEncoder &&source) = delete;

    PhantomCKKSEncoder &operator=(const PhantomCKKSEncoder &assign) = delete;

    PhantomCKKSEncoder &operator=(PhantomCKKSEncoder &&assign) = delete;

    ~PhantomCKKSEncoder() = default;

    template<class T>
    inline void encode(const PhantomContext &context,
                       const std::vector<T> &values,
                       double scale,
                       PhantomPlaintext &destination,
                       size_t chain_index = 1, // first chain index
                       const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        const auto &s = stream_wrapper.get_stream();
        destination.chain_index_ = 0;
        destination.resize(context.coeff_mod_size_, context.poly_degree_, s);
        encode_internal(context, values.data(), values.size(), chain_index, scale, destination, s);
    }

    template<class T>
    [[nodiscard]] inline auto encode(const PhantomContext &context, const std::vector<T> &values,
                                     double scale,
                                     size_t chain_index = 1, // first chain index
                                     const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        PhantomPlaintext destination;
        encode(context, values, scale, destination, chain_index, stream_wrapper);
        return destination;
    }

    template<class T>
    inline void decode(const PhantomContext &context,
                       const PhantomPlaintext &plain,
                       std::vector<T> &destination,
                       const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        const auto &s = stream_wrapper.get_stream();
        auto decoding_sparse_slots = decoding_sparse_slots_ == 0 ? sparse_slots_ : decoding_sparse_slots_;
        if (sparse_slots_ == 0) {
            sparse_slots_ = decoding_sparse_slots_;
        } // temp fix because local encode will change sparse_slots_ but default is 0
        destination.resize(decoding_sparse_slots);
        decode_internal(context, plain, destination.data(), s);
    }
    // inline void cuda_check(cudaError_t e, const char* msg) {
    //     if (e != cudaSuccess) {
    //         throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    //     }
    // }
    // template<class T>
    // inline void decode(const PhantomContext &context,
    //                 const PhantomPlaintext &plain,
    //                 std::vector<T> &destination,
    //                 const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream)
    // {
    //     const auto &s = stream_wrapper.get_stream();

    //     // 1) 计算正确槽位数
    //     size_t slots = 65536 / 2;
    //     // 若你确实有“稀疏解码”，仅当非 0 时才使用
    //     if (decoding_sparse_slots_ != 0) slots = decoding_sparse_slots_;
    //     else if (sparse_slots_ != 0)     slots = sparse_slots_;

    //     if (slots == 0) {                 // 防御：不要调 decode_internal
    //         destination.clear();
    //         return;
    //     }

    //     // 2) 主机输出空间
    //     destination.resize(slots);
    

    //     // 3) 设备端缓冲（复数）
    //     cuDoubleComplex* d_out = nullptr;
    //     cuda_check(cudaMalloc(&d_out, slots * sizeof(cuDoubleComplex)), "cudaMalloc d_out failed");

    //     // 4) 真正的 GPU 解码：传设备指针
    //     decode_internal(context, plain, d_out, s);


    //     // 5) 回拷并写入 destination（T=double 取实部；T=std::complex<double> 直接拷）
    //     std::vector<cuDoubleComplex> h_tmp(slots);
    //     cuda_check(cudaMemcpyAsync(h_tmp.data(), d_out, slots * sizeof(cuDoubleComplex),
    //                             cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync d2h failed");
    //     cuda_check(cudaStreamSynchronize(s), "cudaStreamSynchronize failed");

    //     if constexpr (std::is_same_v<T, double>) {
    //         for (size_t i = 0; i < slots; ++i) destination[i] = h_tmp[i].x; // 取实部
    //     } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    //         for (size_t i = 0; i < slots; ++i) destination[i] = {h_tmp[i].x, h_tmp[i].y};
    //     } else {
    //         static_assert(!sizeof(T*), "Unsupported T for CKKS decode");
    //     }

    //     cudaFree(d_out);
    // }


    inline void decode(const PhantomContext &context,
                       const PhantomPlaintext &plain,
                       std::vector<std::complex<double>> &destination,
                       const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        const auto &s = stream_wrapper.get_stream();
        auto decoding_sparse_slots = decoding_sparse_slots_ == 0 ? sparse_slots_ : decoding_sparse_slots_;
        destination.resize(decoding_sparse_slots);

        std::vector<cuDoubleComplex> output(decoding_sparse_slots);
        decode_internal(context, plain, output.data(), s);

        for (size_t i = 0; i < decoding_sparse_slots; i++) {
            destination[i] = std::complex<double>(output[i].x, output[i].y);
        }
        output.clear();
    }

    template<class T>
    [[nodiscard]] inline auto decode(const PhantomContext &context, const PhantomPlaintext &plain,
                                     const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        std::vector<T> destination;
        decode(context, plain, destination, stream_wrapper);
        return destination;
    }

    [[nodiscard]] inline std::size_t slot_count() const noexcept {
        return slots_;
    }

    // TODO: we may not need this
    // Newly added to provide information about the length of additional messages
    // allowed for encoding after the first call to `encode`
    [[nodiscard]] inline std::size_t message_length() const noexcept {
        if (sparse_slots_ == 0) return slots_;
        return sparse_slots_;
    }

    auto &gpu_ckks_msg_vec() {
        return *gpu_ckks_msg_vec_;
    }

    void reset_sparse_slots() {
        sparse_slots_ = 0;
    }
};
