#pragma once
#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <vector>

using std::vector;

inline vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre(
    vector<PhantomCiphertext> &enc_X,
    const vector<vector<double>> &W,
    int col_X, int col_W, int row_W,
    PhantomContext &context)
{
  vector<PhantomCiphertext> output(static_cast<size_t>(col_W));

  // —— 基本检查 —— //
  if (enc_X.empty() || W.empty() || col_W <= 0 || row_W <= 0)
  {
    std::cout << "ERROR: empty inputs or bad dimensions.\n";
  }
  if (static_cast<int>(enc_X.size()) < row_W)
  {
    std::cout << "ERROR: enc_X size < row_W.\n";
    return output;
  }
  if (static_cast<int>(W.size()) < row_W)
  {
    std::cout << "ERROR: W rows < row_W.\n";
    return output;
  }
  for (int r = 0; r < row_W; ++r)
  {
    if (static_cast<int>(W[r].size()) < col_W)
    {
      std::cout << "ERROR: W row " << r << " has fewer than col_W columns.\n";
      return output;
    }
  }
  if (col_X != row_W)
  {
    std::cout << "ERROR: bad dimensions of X or W.\n";
    return output;
  }
  // cudaSetDevice(1);
  const double scale = enc_X[0].scale();

  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 32));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    }
  }
  if (nthreads == 1)
  {
    stream_pool[0] = *phantom::util::global_variables::default_stream;
  }

  vector<double> time(col_W, 0.0);

#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];
    // phantom::util::cuda_stream_wrapper stream;
    // if (nthreads == 1){
    //   stream = *phantom::util::global_variables::default_stream;
    // } else {
    //   stream = stream_pool[tid];
    // }

    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < col_W; ++i)
    {

      // acc = enc_X[0] * W[0][i]
      PhantomPlaintext ecd_w_0_i;
      // encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
      encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i, stream);
      bridge_to_default(stream); // bridge to default stream after each encode

      PhantomCiphertext acc;
      start = chrono::high_resolution_clock::now();
      evaluator_local.multiply_plain(enc_X[0], ecd_w_0_i, acc);
      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;

      for (int j = 1; j < row_W; ++j)
      {
        PhantomPlaintext ecd_w_j_i;
        // encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
        encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream);
        bridge_to_default(stream); // bridge to default stream after each encode

        PhantomCiphertext tmp;
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain(enc_X[j], ecd_w_j_i, tmp);

        evaluator_local.add_inplace(acc, tmp);
        end = chrono::high_resolution_clock::now();

        elapsed = duration_cast<duration<double>>(end - start).count();

        time[i] += elapsed;
      }

      start = chrono::high_resolution_clock::now();
      evaluator_local.rescale_to_next_inplace(acc, stream);
      acc.scale() = scale;

      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;

      output[static_cast<size_t>(i)] = std::move(acc);
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  // cudaDeviceSynchronize();
  // stream_pool.clear();

  return output;
}

// single thread version
inline vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_single(
    vector<PhantomCiphertext> &enc_X,
    const vector<vector<double>> &W,
    int col_X, int col_W, int row_W,
    PhantomContext &context)
{
  vector<PhantomCiphertext> output(static_cast<size_t>(col_W));

  if (enc_X.empty() || W.empty() || col_W <= 0 || row_W <= 0)
  {
    std::cout << "ERROR: empty inputs or bad dimensions.\n";
  }
  if (static_cast<int>(enc_X.size()) < row_W)
  {
    std::cout << "ERROR: enc_X size < row_W.\n";
    return output;
  }
  if (static_cast<int>(W.size()) < row_W)
  {
    std::cout << "ERROR: W rows < row_W.\n";
    return output;
  }
  for (int r = 0; r < row_W; ++r)
  {
    if (static_cast<int>(W[r].size()) < col_W)
    {
      std::cout << "ERROR: W row " << r << " has fewer than col_W columns.\n";
      return output;
    }
  }
  if (col_X != row_W)
  {
    std::cout << "ERROR: bad dimensions of X or W.\n";
    return output;
  }
  // cudaSetDevice(1);
  const double scale = enc_X[0].scale();

  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 1));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  // if (stream_pool.size() < static_cast<size_t>(nthreads))
  // {
  //   stream_pool.reserve(nthreads);
  //   for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
  //   {
  //     stream_pool.emplace_back();
  //   }
  // }
  // if (nthreads == 1)
  // {
  //   stream_pool[0] = *phantom::util::global_variables::default_stream;
  // }

  vector<double> time(col_W, 0.0);

#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];
    // phantom::util::cuda_stream_wrapper stream;
    // if (nthreads == 1){
    //   stream = *phantom::util::global_variables::default_stream;
    // } else {
    //   stream = stream_pool[tid];
    // }

    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < col_W; ++i)
    {

      // acc = enc_X[0] * W[0][i]
      PhantomPlaintext ecd_w_0_i;
      // encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
      encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
      bridge_to_default(stream); // bridge to default stream after each encode

      PhantomCiphertext acc;
      start = chrono::high_resolution_clock::now();
      evaluator_local.multiply_plain(enc_X[0], ecd_w_0_i, acc);
      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;

      for (int j = 1; j < row_W; ++j)
      {
        PhantomPlaintext ecd_w_j_i;
        // encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
        encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
        // bridge_to_default(stream); // bridge to default stream after each encode

        PhantomCiphertext tmp;
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain(enc_X[j], ecd_w_j_i, tmp);
        evaluator_local.add_inplace(acc, tmp);
        end = chrono::high_resolution_clock::now();

        elapsed = duration_cast<duration<double>>(end - start).count();

        time[i] += elapsed;
      }

      start = chrono::high_resolution_clock::now();
      evaluator_local.rescale_to_next_inplace(acc);
      acc.scale() = scale;

      end = chrono::high_resolution_clock::now();
      elapsed = duration_cast<duration<double>>(end - start).count();
      // cout << "[DEBUG] time: " << elapsed << endl;
      time[i] += elapsed;

      output[static_cast<size_t>(i)] = std::move(acc);
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  // cudaDeviceSynchronize();
  // stream_pool.clear();

  return output;
}

// vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre(vector<PhantomCiphertext> & enc_X,
//   vector<vector<double>> & W, int col_X, int col_W, int row_W,
//   PhantomContext& context){

//   // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   // const auto &stream = stream_wrapper.get_stream();

//   vector<PhantomCiphertext> output(col_W);
//   double scale = enc_X[0].scale();

//   if(col_X != row_W){
//     cout <<"ERROR: bad dimensions of X or W. "<<endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(context);
//   //pack Phantom to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//     // #pragma acc parallel loop
//     for (int i = 0; i < col_W; ++i){
//         // cout << "outer loop <<< " << i << endl;
//         // cout << "========================================================" << endl;
//         //encode w[0][i]
//         PhantomPlaintext ecd_w_0_i;
//         // std::chrono::_V2::system_clock::time_point start_encode = high_resolution_clock::now();
//         encoder.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i); // chain index is default 1
//         // std::chrono::_V2::system_clock::time_point end_encode = high_resolution_clock::now();
//         // std::chrono::duration<double, std::milli> encode_duration = end_encode - start_encode;
//         // cout << "[DEBUG] Encoding duration for W[0][" << i << "]: " << encode_duration.count() << " ms" << endl;

//         // std::chrono::_V2::system_clock::time_point start_mul = high_resolution_clock::now();
//         //enc_X[0]*ecd_w[0][i]
//         evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i]);
//         //evaluator.rescale_to_next_inplace(output[i]);
//         // std::chrono::_V2::system_clock::time_point end_mul = high_resolution_clock::now();
//         // std::chrono::duration<double, std::milli> mul_duration = end_mul - start_mul;
//         // cout << "[DEBUG] Multiplication duration for W[0][" << i << "]: " << mul_duration.count() << " ms" << endl;

//         for (int j = 1 ; j < row_W ; ++j){
//           // cout << "inter loop <<< " << j << endl;
//           // cout << "-----------------------------------------------------" << endl;
//           //encode w[j][i]
//           PhantomPlaintext ecd_w_j_i;

//           // std::chrono::_V2::system_clock::time_point start_encode = high_resolution_clock::now();
//           encoder.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
//           // std::chrono::_V2::system_clock::time_point end_encode = high_resolution_clock::now();
//           // std::chrono::duration<double, std::milli> encode_duration = end_encode - start_encode;
//           // cout << "[DEBUG] Encoding duration for W[" << j << "][" << i << "]: " << encode_duration.count() << " ms" << endl;

//           //enc_X[j]*ecd_w[j][i]
//           PhantomCiphertext temp;
//           // std::chrono::_V2::system_clock::time_point start_mul = high_resolution_clock::now();
//           evaluator.multiply_plain(enc_X[j], ecd_w_j_i, temp);
//           // std::chrono::_V2::system_clock::time_point end_mul = high_resolution_clock::now();
//           // std::chrono::duration<double, std::milli> mul_duration = end_mul - start_mul;
//           // cout << "[DEBUG] Multiplication duration for W[" << j << "][" << i << "]: " << mul_duration.count() << " ms" << endl;

//           //evaluator.rescale_to_next_inplace(temp);
//           // std::chrono::_V2::system_clock::time_point start_add = high_resolution_clock::now();
//           evaluator.add_inplace(output[i],temp);
//           // std::chrono::_V2::system_clock::time_point end_add = high_resolution_clock::now();
//           // std::chrono::duration<double, std::milli> add_duration = end_add - start_add;
//           // cout << "[DEBUG] Addition duration for W[" << j << "][" << i << "]: " << add_duration.count() << " ms" << endl;
//         }

//         evaluator.rescale_to_next_inplace(output[i]);
//         output[i].scale()=scale;
//     }

//   return output;

// }

// vector<PhantomCiphertext> ct_pt_matrix_mul(vector<PhantomCiphertext> &enc_X,
//                                            vector<vector<PhantomPlaintext>> &W, int col_X, int col_W, int row_W,
//                                            PhantomContext &context)
// {

//   vector<PhantomCiphertext> output(col_W);

//   if (col_X != row_W)
//   {
//     cout << "ERROR: bad dimensions of X or W. " << endl;
//     return output;
//   }

//   // CKKSEncoder encoder(seal_context);
//   // Evaluator evaluator(seal_context, encoder);
//   PhantomCKKSEncoder phantom_encoder(context);
//   // pack Phantom to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   // #pragma omp parallel for

//   for (int i = 0; i < col_W; ++i)
//   {

//     // encode w[0][i]
//     // Plaintext ecd_w_0_i;
//     // encoder.encode(W[0][i], scale, ecd_w_0_i);

//     // enc_X[0]*ecd_w[0][i]
//     evaluator.multiply_plain(enc_X[0], W[0][i], output[i]);
//     // evaluator.rescale_to_next_inplace(output[i]);

//     for (int j = 1; j < row_W; ++j)
//     {
//       // encode w[j][i]
//       // Plaintext ecd_w_j_i;
//       // encoder.encode(W[j][i], scale, ecd_w_j_i);

//       // enc_X[j]*ecd_w[j][i]
//       PhantomCiphertext temp;
//       evaluator.multiply_plain(enc_X[j], W[j][i], temp);
//       // evaluator.rescale_to_next_inplace(temp);
//       evaluator.add_inplace(output[i], temp);
//     }

//     evaluator.rescale_to_next_inplace(output[i]);
//   }

//   return output;
// }

// CUDA_API_PER_THREAD_DEFAULT_STREAM=1 for testing default stream isolation

vector<PhantomCiphertext> ct_pt_matrix_mul(vector<PhantomCiphertext> &enc_X,
                                           vector<vector<PhantomPlaintext>> &W, int col_X, int col_W, int row_W,
                                           PhantomContext &context)
{

  // vector<PhantomCiphertext> output(col_W);
  vector<PhantomCiphertext> output(static_cast<size_t>(col_W));

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }


  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, col_W));
  // // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    }
  }
  if (nthreads == 1)
  {
    stream_pool[0] = *phantom::util::global_variables::default_stream;
  }

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  cudaStream_t timing_stream = nullptr;
  cudaStreamCreateWithFlags(&timing_stream, cudaStreamNonBlocking);
  // CKKSEncoder encoder(seal_context);
  // Evaluator evaluator(seal_context, encoder);
  // PhantomCKKSEncoder phantom_encoder(context);
  // pack Phantom to SEAL style
  // Encoder encoder(&context, &phantom_encoder);
  // Evaluator evaluator(&context, &phantom_encoder);

  // static std::mutex x_locks;
// #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];

    // std::vector<PhantomCiphertext> X_local(row_W);
    // for (int j = 0; j < row_W; ++j)
    //   X_local[j] = enc_X[j];
    std::vector<PhantomCiphertext> X_local(static_cast<size_t>(row_W));
    for (int j = 0; j < row_W; ++j)
    {
      X_local[static_cast<size_t>(j)] = deep_copy_cipher(enc_X[j], context, stream);
    }

    cudaStreamWaitEvent(stream.get_stream(), ev_start, 0);

#pragma omp barrier
#pragma omp single
    {
      cudaEventRecord(ev_start, timing_stream ? timing_stream : 0);
    }

#pragma omp for schedule(static)
    for (int i = 0; i < col_W; ++i)
    {
      // PhantomCiphertext x0 = deep_copy_cipher(enc_X[0], context, stream);
      // PhantomPlaintext p0 = W[0][i];
      PhantomCiphertext acc, temp;
      // encode w[0][i]
      // Plaintext ecd_w_0_i;
      // encoder.encode(W[0][i], scale, ecd_w_0_i);

      // enc_X[0]*ecd_w[0][i]
      // std::lock_guard<std::mutex> g(x_locks);
      evaluator_local.multiply_plain(X_local[0], W[0][i], acc, stream);
      // evaluator_local.multiply_plain(x0, p0, acc, stream);
      // evaluator_local.multiply_plain(enc_X[0], W[0][i], acc, stream);
      // bridge_to_default(stream);
      // evaluator.rescale_to_next_inplace(output[i]);

      for (int j = 1; j < row_W; ++j)
      {
        // PhantomCiphertext xj = deep_copy_cipher(enc_X[j], context, stream);
        // PhantomPlaintext pj = W[j][i];
        // encode w[j][i]
        // Plaintext ecd_w_j_i;
        // encoder.encode(W[j][i], scale, ecd_w_j_i);

        // enc_X[j]*ecd_w[j][i]
        // PhantomCiphertext temp;
        // evaluator_local.multiply_plain(enc_X[j], W[j][i], temp, stream);
        // evaluator_local.multiply_plain(xj, pj, temp, stream);
        evaluator_local.multiply_plain(X_local[j], W[j][i], temp, stream);
        // evaluator.rescale_to_next_inplace(temp);
        evaluator_local.add_inplace(acc, temp, stream);
      }

      evaluator_local.rescale_to_next_inplace(acc, stream);
      // bridge_to_default(stream);
      // cudaStreamSynchronize(stream.get_stream());
      output[static_cast<size_t>(i)] = std::move(acc);
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  std::vector<cudaEvent_t> ev_done(nthreads);
  for (int i = 0; i < nthreads; ++i)
  {
    cudaEventCreateWithFlags(&ev_done[i], cudaEventDisableTiming);
    cudaEventRecord(ev_done[i], stream_pool[i].get_stream());
  }

  for (int i = 0; i < nthreads; ++i)
  {
    cudaStreamWaitEvent(timing_stream ? timing_stream : 0, ev_done[i], 0);
  }

  cudaEventRecord(ev_stop, timing_stream ? timing_stream : 0);
  cudaEventSynchronize(ev_stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  cout << "Ct-Pt compute time = " << ms << " ms\n";

  for (auto &e : ev_done)
    cudaEventDestroy(e);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  if (timing_stream)
    cudaStreamDestroy(timing_stream);

  cudaDeviceSynchronize();
  return output;
}

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_large(vector<PhantomCiphertext> &enc_X,
                                                        vector<vector<double>> &W, int col_X, int col_W, int row_W,
                                                        PhantomContext &context)
{

  // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  // const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W);
  double scale = enc_X[0].scale();

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  int col_W_t = col_W / 128;

  // #pragma omp parallel for
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 32));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    }
  }

  vector<double> time(128, 0.0);
#pragma omp parallel num_threads(nthreads)
  {
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];
    // phantom::util::cuda_stream_wrapper stream;
    // if (nthreads == 1){
    //   stream = *phantom::util::global_variables::default_stream;
    // } else {
    //   stream = stream_pool[tid];
    // }

    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < 128; ++i)
    {
      for (int k = 0; k < col_W_t; ++k)
      {
        // encode w[0][i]
        PhantomPlaintext ecd_w_0_i;
        // encoder_local.encode(W[0][i * col_W_t + k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        encoder_local.encode(W[0][i * col_W_t + k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i, stream);
        bridge_to_default(stream); // bridge to default stream after each encode
        // enc_X[0]*ecd_w[0][i]
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        time[i] += elapsed;
        // evaluator.rescale_to_next_inplace(output[i]);

        for (int j = 1; j < row_W; ++j)
        {
          // encode w[j][i]
          PhantomPlaintext ecd_w_j_i;
          // encoder_local.encode(W[j][i * col_W_t + k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
          encoder_local.encode(W[j][i * col_W_t + k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream);
          bridge_to_default(stream); // brige to default stream after each encode

          // enc_X[j]*ecd_w[j][i]
          PhantomCiphertext temp;
          start = chrono::high_resolution_clock::now();
          evaluator_local.multiply_plain(enc_X[j], ecd_w_j_i, temp);
          // if(i == 0)cout <<log2(temp.scale())<<" "<<log2(output[i*col_W_t+k].scale())<<endl;
          // evaluator.rescale_to_next_inplace(temp);
          evaluator_local.add_inplace(output[i * col_W_t + k], temp);
          end = chrono::high_resolution_clock::now();
          elapsed = duration_cast<duration<double>>(end - start).count();
          time[i] += elapsed;
        }

        evaluator_local.rescale_to_next_inplace(output[i * col_W_t + k], stream);
        output[i * col_W_t + k].scale() = scale;
        // if(i == 0) cout <<log2(output[i*col_W_t+k].scale())<<endl;
      }
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  return output;
}

// vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_large(vector<PhantomCiphertext> &enc_X,
//                                                         vector<vector<double>> &W, int col_X, int col_W, int row_W,
//                                                         PhantomContext &context)
// {

//   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   const auto &stream = stream_wrapper.get_stream();

//   vector<PhantomCiphertext> output(col_W);
//   double scale = enc_X[0].scale();

//   if (col_X != row_W)
//   {
//     cout << "ERROR: bad dimensions of X or W. " << endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   int col_W_t = col_W / 128;

//   // #pragma omp parallel for

//   for (int i = 0; i < 128; ++i)
//   {
//     for (int k = 0; k < col_W_t; ++k)
//     {
//       // encode w[0][i]
//       PhantomPlaintext ecd_w_0_i;
//       encoder.encode(W[0][i * col_W_t + k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
//       // enc_X[0]*ecd_w[0][i]
//       evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
//       // evaluator.rescale_to_next_inplace(output[i]);

//       for (int j = 1; j < row_W; ++j)
//       {
//         // encode w[j][i]
//         PhantomPlaintext ecd_w_j_i;
//         encoder.encode(W[j][i * col_W_t + k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);

//         // enc_X[j]*ecd_w[j][i]
//         PhantomCiphertext temp;
//         evaluator.multiply_plain(enc_X[j], ecd_w_j_i, temp);
//         // if(i == 0)cout <<log2(temp.scale())<<" "<<log2(output[i*col_W_t+k].scale())<<endl;
//         // evaluator.rescale_to_next_inplace(temp);
//         evaluator.add_inplace(output[i * col_W_t + k], temp);
//       }

//       evaluator.rescale_to_next_inplace(output[i * col_W_t + k]);
//       output[i * col_W_t + k].scale() = scale;
//       // if(i == 0) cout <<log2(output[i*col_W_t+k].scale())<<endl;
//     }
//   }

//   return output;
// }

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_w_mask(vector<PhantomCiphertext> &enc_X,
                                                         vector<vector<double>> &W, const vector<int> &bias_vec, int col_X, int col_W, int row_W,
                                                         PhantomContext &context)
{

  // const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  // const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W);
  double scale = enc_X[0].scale();

  if (col_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  size_t slot_count = phantom_encoder.slot_count();

  int col_W_t = col_W / 128;
  // cout <<col_W_t<<endl;

  // #pragma omp parallel for
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 32));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back();
    }
  }

  vector<double> time(128, 0.0);

#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];
    // phantom::util::cuda_stream_wrapper stream;
    // if (nthreads == 1){
    //   stream = *phantom::util::global_variables::default_stream;
    // } else {
    //   stream = stream_pool[tid];
    // }

    chrono::high_resolution_clock::time_point start, end;
    double elapsed;

#pragma omp for schedule(static)
    for (int i = 0; i < 128; ++i)
    {
      for (int k = 0; k < col_W_t; ++k)
      {
        // encode w[0][i]
        vector<double> temp(slot_count, 0);
        for (int j = 0; j < slot_count; ++j)
        {
          if (bias_vec[j] == 1)
          {
            temp[j] = W[0][i * col_W_t + k];
          }
        }
        PhantomPlaintext ecd_w_0_i;
        // encoder_local.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        encoder_local.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i, stream);
        bridge_to_default(stream); // brige to default stream after each encode
        // enc_X[0]*ecd_w[0][i]
        start = chrono::high_resolution_clock::now();
        evaluator_local.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
        end = chrono::high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start).count();
        // cout << "[DEBUG] time: " << elapsed << endl;
        time[i] += elapsed;
        // evaluator.rescale_to_next_inplace(output[i]);
        // cout <<"mul 1."<<endl;

        for (int j = 1; j < row_W; ++j)
        {
          // cout <<j<<" ";
          // encode w[j][i]
          vector<double> tempw(slot_count, 0);
          for (int kk = 0; kk < slot_count; ++kk)
          {
            if (bias_vec[kk] == 1)
            {
              tempw[kk] = W[j][i * col_W_t + k];
            }
          }
          PhantomPlaintext ecd_w_j_i;
          // encoder_local.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);
          encoder_local.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream);
          bridge_to_default(stream); // brige to default stream after each encode
          // enc_X[j]*ecd_w[j][i]
          PhantomCiphertext tempx;
          start = chrono::high_resolution_clock::now();
          evaluator_local.multiply_plain(enc_X[j], ecd_w_j_i, tempx);

          // cout <<"mul. "<<endl;
          // evaluator.rescale_to_next_inplace(temp);
          evaluator_local.add_inplace(output[i * col_W_t + k], tempx);
          end = chrono::high_resolution_clock::now();
          elapsed = duration_cast<duration<double>>(end - start).count();
          time[i] += elapsed;
          // cout <<"add. "<<endl;
        }

        evaluator_local.rescale_to_next_inplace(output[i * col_W_t + k], stream);
        output[i * col_W_t + k].scale() = scale;
      }
    }
    cudaStreamSynchronize(stream.get_stream());
  }
  // cout <<log(output[0].scale())<<endl;

  // double total = 0.0;
  // for (double t : time){
  //   total += t;
  // }
  // cout << "ct-pt time(without encoding)" << total << " s" << endl;

  return output;
}

// vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_w_mask(vector<PhantomCiphertext> &enc_X,
//                                                          vector<vector<double>> &W, const vector<int> &bias_vec, int col_X, int col_W, int row_W,
//                                                          PhantomContext &context)
// {

//   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   const auto &stream = stream_wrapper.get_stream();

//   vector<PhantomCiphertext> output(col_W);
//   double scale = enc_X[0].scale();

//   if (col_X != row_W)
//   {
//     cout << "ERROR: bad dimensions of X or W. " << endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   size_t slot_count = phantom_encoder.slot_count();

//   int col_W_t = col_W / 128;
//   // cout <<col_W_t<<endl;

//   // #pragma omp parallel for

//   for (int i = 0; i < 128; ++i)
//   {
//     for (int k = 0; k < col_W_t; ++k)
//     {
//       // encode w[0][i]
//       vector<double> temp(slot_count, 0);
//       for (int j = 0; j < slot_count; ++j)
//       {
//         if (bias_vec[j] == 1)
//         {
//           temp[j] = W[0][i * col_W_t + k];
//         }
//       }
//       PhantomPlaintext ecd_w_0_i;
//       encoder.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
//       // enc_X[0]*ecd_w[0][i]
//       evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i * col_W_t + k]);
//       // evaluator.rescale_to_next_inplace(output[i]);
//       // cout <<"mul 1."<<endl;

//       for (int j = 1; j < row_W; ++j)
//       {
//         // cout <<j<<" ";
//         // encode w[j][i]
//         vector<double> tempw(slot_count, 0);
//         for (int kk = 0; kk < slot_count; ++kk)
//         {
//           if (bias_vec[kk] == 1)
//           {
//             tempw[kk] = W[j][i * col_W_t + k];
//           }
//         }
//         PhantomPlaintext ecd_w_j_i;
//         encoder.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);

//         // enc_X[j]*ecd_w[j][i]
//         PhantomCiphertext tempx;
//         evaluator.multiply_plain(enc_X[j], ecd_w_j_i, tempx);
//         // cout <<"mul. "<<endl;
//         // evaluator.rescale_to_next_inplace(temp);
//         evaluator.add_inplace(output[i * col_W_t + k], tempx);
//         // cout <<"add. "<<endl;
//       }

//       evaluator.rescale_to_next_inplace(output[i * col_W_t + k]);
//       output[i * col_W_t + k].scale() = scale;
//     }
//   }
//   // cout <<log(output[0].scale())<<endl;

//   return output;
// }

// vector<PhantomCiphertext> ct_pt_matrix_mul(const vector<PhantomCiphertext> & enc_X,
//   const vector<vector<PhantomPlaintext>> & W, int col_X, int col_W, int row_W,
//   PhantomContext& seal_context){

//   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
//   const auto &stream = stream_wrapper.get_stream();
//   vector<PhantomCiphertext> output(col_W);

//   if(col_X != row_W){
//     cout <<"ERROR: bad dimensions of X or W. "<<endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(seal_context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   // #pragma omp parallel for

//   for (int i = 0; i < col_W; ++i){

//     //encode w[0][i]
//     //Plaintext ecd_w_0_i;
//     //encoder.encode(W[0][i], scale, ecd_w_0_i);

//     //enc_X[0]*ecd_w[0][i]
//     evaluator.multiply_plain(enc_X[0], W[0][i], output[i]);
//     //evaluator.rescale_to_next_inplace(output[i]);

//     for (int j = 1 ; j < row_W ; ++j){
//       //encode w[j][i]
//      // Plaintext ecd_w_j_i;
//      // encoder.encode(W[j][i], scale, ecd_w_j_i);

//       //enc_X[j]*ecd_w[j][i]
//       PhantomCiphertext temp;
//       evaluator.multiply_plain(enc_X[j], W[j][i], temp);
//       //evaluator.rescale_to_next_inplace(temp);
//       evaluator.add_inplace(output[i],temp);
//     }

//     evaluator.rescale_to_next_inplace(output[i]);

//   }

//   return output;

// }