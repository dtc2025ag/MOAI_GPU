#pragma once
#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

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

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <vector>

// 假定以下类型/接口已在你的工程里声明：
// PhantomContext, PhantomCiphertext, PhantomPlaintext,
// PhantomCKKSEncoder, Encoder, Evaluator

using std::vector;
using phantom::util::cuda_stream_wrapper;


vector<cuda_stream_wrapper> stream_pool; // 线程私有流池

// 把“自定义流上的完成”桥接到默认流，避免默认流读到半成品
static inline void bridge_to_default(const cuda_stream_wrapper &sw) {
  auto dst = phantom::util::global_variables::default_stream->get_stream();
  if (sw.get_stream() == dst) return;                 // 同一条流就不需要桥接

  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaEventRecord(ev, sw.get_stream());               // 在生产流上记录事件
  cudaStreamWaitEvent(dst, ev, 0);                    // 让“库的默认流”等待事件
  cudaEventDestroy(ev);                               // 等待已入队，销毁事件对象即可
}

inline vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre(
    vector<PhantomCiphertext> &enc_X,              // ★ 非 const：匹配 evaluator 的非 const & 接口
    const vector<vector<double>> &W,               // 权重只读即可
    int col_X, int col_W, int row_W,
    PhantomContext &context)
{
  vector<PhantomCiphertext> output(static_cast<size_t>(col_W));

  // —— 基本检查 —— //
  if (enc_X.empty() || W.empty() || col_W <= 0 || row_W <= 0) {
    std::cout << "ERROR: empty inputs or bad dimensions.\n";
    return output;
  }
  if (static_cast<int>(enc_X.size()) < row_W) {
    std::cout << "ERROR: enc_X size < row_W.\n";
    return output;
  }
  if (static_cast<int>(W.size()) < row_W) {
    std::cout << "ERROR: W rows < row_W.\n";
    return output;
  }
  for (int r = 0; r < row_W; ++r) {
    if (static_cast<int>(W[r].size()) < col_W) {
      std::cout << "ERROR: W row " << r << " has fewer than col_W columns.\n";
      return output;
    }
  }
  if (col_X != row_W) {
    std::cout << "ERROR: bad dimensions of X or W.\n";
    return output;
  }

  const double scale = enc_X[0].scale();

  // 线程数：不超过列数（避免空转）
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, col_W));
  // std::cout << "nums of thread: " << nthreads << std::endl;

  // —— 准备每线程一个流（拥有型 wrapper） —— //
  if (nthreads == 1) {
    stream_pool.reserve(nthreads);
    stream_pool[0] = *phantom::util::global_variables::default_stream;
  }
  else if (stream_pool.size() < static_cast<size_t>(nthreads)) {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i) {
      stream_pool.emplace_back(); // 默认构造：内部创建并持有一个新 CUDA 流
    }
  }

  // —— 并行计算：每线程独立 Encoder/Evaluator（各自绑定线程私有的 PhantomCKKSEncoder） —— //
  #pragma omp parallel num_threads(nthreads)
  {
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder   encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid];              // ★ 引用，不要拷贝 wrapper

    #pragma omp for schedule(static)
    for (int i = 0; i < col_W; ++i) {
      // acc = enc_X[0] * W[0][i]
      PhantomPlaintext ecd_w_0_i;
      encoder_local.encode(W[0][i], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i, stream);
      bridge_to_default(stream);                  // ★ 跨流桥接


      PhantomCiphertext acc;
      evaluator_local.multiply_plain(enc_X[0], ecd_w_0_i, acc /* 在默认流上 */);

      // 逐行累加 enc_X[j] * W[j][i]

      for (int j = 1; j < row_W; ++j) {
        PhantomPlaintext ecd_w_j_i;
        encoder_local.encode(W[j][i], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i, stream);
        bridge_to_default(stream);                // ★ 每次 encode 后桥接

        PhantomCiphertext tmp;
        evaluator_local.multiply_plain(enc_X[j], ecd_w_j_i, tmp /* 默认流 */);

        // 如需对齐 level，可按需打开
        // if (tmp.params_id() != acc.params_id()) {
        //     evaluator_local.mod_switch_to_inplace(tmp, acc.params_id());
        // }

        evaluator_local.add_inplace(acc, tmp /* 默认流 */);
      }

    // 是否 rescale 取决于你的算术设计；这里保持和你原逻辑一致
      evaluator_local.rescale_to_next_inplace(acc /* 默认流 */);
      acc.scale() = scale;

      // 放回结果
      output[static_cast<size_t>(i)] = std::move(acc);
      



    }
  }

  // 保守起见：并行段后做一次全局同步，确保所有默认流工作完成
  cudaDeviceSynchronize();

  return output;
}


vector<PhantomCiphertext> ct_pt_matrix_mul(vector<PhantomCiphertext> & enc_X, 
  vector<vector<PhantomPlaintext>> & W, int col_X, int col_W, int row_W, 
  PhantomContext& context){

  vector<PhantomCiphertext> output(col_W);

  if(col_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  // CKKSEncoder encoder(seal_context);
  // Evaluator evaluator(seal_context, encoder);
  PhantomCKKSEncoder phantom_encoder(context);
  //pack Phantom to SEAL style
  Encoder encoder(&context, &phantom_encoder); 
  Evaluator evaluator(&context, &phantom_encoder);

  // #pragma omp parallel for 

  for (int i = 0; i < col_W; ++i){

    //encode w[0][i]
    //Plaintext ecd_w_0_i;
    //encoder.encode(W[0][i], scale, ecd_w_0_i);

    //enc_X[0]*ecd_w[0][i]
    evaluator.multiply_plain(enc_X[0], W[0][i], output[i]);
    //evaluator.rescale_to_next_inplace(output[i]);

    for (int j = 1 ; j < row_W ; ++j){
      //encode w[j][i]
     // Plaintext ecd_w_j_i;
     // encoder.encode(W[j][i], scale, ecd_w_j_i);

      //enc_X[j]*ecd_w[j][i]
      PhantomCiphertext temp;
      evaluator.multiply_plain(enc_X[j], W[j][i], temp);
      //evaluator.rescale_to_next_inplace(temp);
      evaluator.add_inplace(output[i],temp);
    }

    evaluator.rescale_to_next_inplace(output[i]);

  }

  return output;

}

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_large(vector<PhantomCiphertext> & enc_X, 
  vector<vector<double>> & W, int col_X, int col_W, int row_W, 
  PhantomContext& context){

  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W);
  double scale = enc_X[0].scale();

  if(col_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

    int col_W_t = col_W/128;

    // #pragma omp parallel for

    for (int i = 0; i < 128; ++i){
      for (int k = 0 ; k < col_W_t ; ++k){
        //encode w[0][i]
        PhantomPlaintext ecd_w_0_i;
        encoder.encode(W[0][i*col_W_t+k], enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
        //enc_X[0]*ecd_w[0][i]
        evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i*col_W_t+k]);
        //evaluator.rescale_to_next_inplace(output[i]);

        for (int j = 1 ; j < row_W ; ++j){
          //encode w[j][i]
          PhantomPlaintext ecd_w_j_i;
          encoder.encode(W[j][i*col_W_t+k], enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);

          //enc_X[j]*ecd_w[j][i]
          PhantomCiphertext temp;
          evaluator.multiply_plain(enc_X[j], ecd_w_j_i, temp);
          //if(i == 0)cout <<log2(temp.scale())<<" "<<log2(output[i*col_W_t+k].scale())<<endl;
          //evaluator.rescale_to_next_inplace(temp);
          evaluator.add_inplace(output[i*col_W_t+k],temp);
        }

        evaluator.rescale_to_next_inplace(output[i*col_W_t+k]);
        output[i*col_W_t+k].scale()=scale;
        //if(i == 0) cout <<log2(output[i*col_W_t+k].scale())<<endl;
      }
    }
  
  return output;

}

vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_w_mask(vector<PhantomCiphertext> & enc_X, 
  vector<vector<double>> & W,const vector<int> & bias_vec, int col_X, int col_W, int row_W, 
  PhantomContext& context){

  const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
  const auto &stream = stream_wrapper.get_stream();

  vector<PhantomCiphertext> output(col_W);
  double scale = enc_X[0].scale();

  if(col_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  size_t slot_count = phantom_encoder.slot_count();

  int col_W_t = col_W/128;
  //cout <<col_W_t<<endl;

  // #pragma omp parallel for 

  for (int i = 0; i < 128; ++i){
    for (int k = 0 ; k < col_W_t ; ++k){
      //encode w[0][i]
      vector<double> temp(slot_count,0);
      for (int j = 0 ; j < slot_count ; ++j){
        if(bias_vec[j] == 1){
          temp[j] = W[0][i*col_W_t+k];
        }
      }
      PhantomPlaintext ecd_w_0_i;
      encoder.encode(temp, enc_X[0].params_id(), enc_X[0].scale(), ecd_w_0_i);
      //enc_X[0]*ecd_w[0][i]
      evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i*col_W_t+k]);
      //evaluator.rescale_to_next_inplace(output[i]);
      //cout <<"mul 1."<<endl;

      for (int j = 1 ; j < row_W ; ++j){
        //cout <<j<<" ";
        //encode w[j][i]
        vector<double> tempw(slot_count,0);
        for (int kk = 0 ; kk < slot_count ; ++kk){
          if(bias_vec[kk] == 1){
            tempw[kk] = W[j][i*col_W_t+k];
          }
        }
        PhantomPlaintext ecd_w_j_i;
        encoder.encode(tempw, enc_X[j].params_id(), enc_X[j].scale(), ecd_w_j_i);

        //enc_X[j]*ecd_w[j][i]
        PhantomCiphertext tempx;
        evaluator.multiply_plain(enc_X[j], ecd_w_j_i, tempx);
        //cout <<"mul. "<<endl;
        //evaluator.rescale_to_next_inplace(temp);
        evaluator.add_inplace(output[i*col_W_t+k],tempx);
        //cout <<"add. "<<endl;
      }

      evaluator.rescale_to_next_inplace(output[i*col_W_t+k]);
      output[i*col_W_t+k].scale()=scale;

    }
  }
  //cout <<log(output[0].scale())<<endl;

  return output;

}

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