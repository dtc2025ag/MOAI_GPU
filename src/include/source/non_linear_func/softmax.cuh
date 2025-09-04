// #include <chrono>
// using namespace chrono;
// #include "Bootstrapper.h"
// #include "ckks_evaluator.h"
#include "include.cuh"
// #include "../bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

PhantomCiphertext exp(PhantomCiphertext &x, PhantomContext &context, PhantomRelinKey &relin_keys, phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
  // CKKSEncoder encoder(context);
  // Evaluator evaluator(context, encoder);
  PhantomCKKSEncoder phantom_encoder(context);
  // repack the phantom encoder to SEAL style
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  PhantomPlaintext inverse_128;
  encoder.encode(0.0078125, x.params_id(), x.scale(), inverse_128, stream);
  // evaluator.mod_switch_to_inplace(inverse_128,x.parms_id());
  // cout <<"encode 0.0078125"<<endl;

  PhantomCiphertext output;
  evaluator.multiply_plain(x, inverse_128, output, stream);
  evaluator.rescale_to_next_inplace(output, stream);
  // cout <<"x*0.0078125"<<endl;
  // cout <<log2(output.scale())<<endl;

  PhantomPlaintext one;
  encoder.encode(1.0, output.params_id(), output.scale(), one, stream);
  // cout <<"encode 1"<<endl;
  // PhantomCiphertext res;
  evaluator.add_plain_inplace(output, one, stream);
  // cout <<"x*0.0078125+1"<<endl;
  // evaluator.rescale_to_next_inplace(output);
  // cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output.parms_id()).chain_depth()<<endl;

  // compute output^128
  for (int i = 0; i < log2(128); ++i)
  {
    // cout <<i<<endl;
    evaluator.square_inplace(output, stream);
    evaluator.relinearize_inplace(output, relin_keys, stream);
    evaluator.rescale_to_next_inplace(output, stream);
  }
  // cout <<"(x*0.0078125+1)^128"<<endl;
  // cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output.parms_id()).chain_depth()<<endl;

  return output;
}

PhantomCiphertext inverse(PhantomCiphertext &x, PhantomContext &context,
                          PhantomRelinKey &relin_keys, int iter, phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
  // by default, iter = 4 (from Nexus)
  //  CKKSEncoder encoder(seal_context);
  //  Evaluator evaluator(seal_context, encoder);
  PhantomCKKSEncoder phantom_encoder(context);
  // repack the phantom encoder to SEAL style
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  PhantomPlaintext one;
  encoder.encode(1.0, x.params_id(), x.scale(), one, stream);

  PhantomCiphertext y;
  evaluator.sub_plain(x, one, y, stream);
  evaluator.negate_inplace(y, stream);

  PhantomCiphertext tmp;
  evaluator.add_plain(y, one, tmp, stream);

  PhantomCiphertext res = tmp;
  for (int i = 0; i < iter; ++i)
  {
    evaluator.square_inplace(y, stream);
    evaluator.relinearize_inplace(y, relin_keys, stream);
    evaluator.rescale_to_next_inplace(y, stream);

    // cout <<"y scale = "<<log2(y.scale())<<" , one scale = "<<log2(one.scale())<<endl;
    encoder.encode(1.0, y.params_id(), y.scale(), one, stream);
    evaluator.add_plain(y, one, tmp, stream);

    evaluator.mod_switch_to_inplace(res, tmp.params_id(), stream);
    evaluator.multiply_inplace(res, tmp, stream);
    evaluator.relinearize_inplace(res, relin_keys, stream);
    evaluator.rescale_to_next_inplace(res, stream);
  }

  return res;
}

vector<PhantomCiphertext> softmax(vector<PhantomCiphertext> &enc_X, vector<int> &bias_vec, int input_num, PhantomContext &context,
                                  PhantomRelinKey &relin_keys, int iter, PhantomSecretKey &sk)
{

  int num = enc_X.size();
  // cout << "number of ct in output = " << num << endl;
  vector<PhantomCiphertext> output(num);

  // PhantomCKKSEncoder encoder(context);
  // Evaluator evaluator(context, encoder);
  PhantomCKKSEncoder phantom_encoder(context);
  // repack the phantom encoder to SEAL style
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  size_t slot_count = encoder.slot_count();
  // for test
  Decryptor decryptor(&context, &sk);
  // int slot_count = encoder.slot_count();
  // cout <<"slot count = "<<slot_count<<endl;
  size_t num_batch = slot_count / 128;
  // cout <<"number of batch = "<<num_batch<<endl;

  // compute x_ij - 8
  vector<PhantomCiphertext> enc_x_minus(num);

  double minus_index = 8.1;
  vector<double> minus(slot_count, 0);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      minus[i] = minus_index;
    }
  }

  for (int i = 0; i < num; ++i)
  {
    enc_x_minus[i] = enc_X[i];
    // for slot with value neq 0, minus 8
    // case 0: first line
    if (i == 0)
    {
      PhantomPlaintext one;
      encoder.encode(minus, enc_x_minus[i].scale(), one);
      evaluator.mod_switch_to_inplace(one, enc_x_minus[i].params_id());
      evaluator.sub_plain_inplace(enc_x_minus[i], one);
    }
    // case1: all zero row
    else if (i > input_num && i <= (num - input_num))
    {
    }
    // case2: 0 - input_num line
    else if (i <= input_num)
    {
      vector<double> temps1(slot_count, 0);
      int index = num_batch * (input_num - i);
      for (int i = 0; i < slot_count; ++i)
      {
        if (bias_vec[i] == 1 && i < index)
        {
          temps1[i] = minus_index;
        }
      }
      // cout << index / num << endl;
      // s1[index] = 1;
      PhantomPlaintext one;
      encoder.encode(temps1, enc_x_minus[i].scale(), one);
      evaluator.mod_switch_to_inplace(one, enc_x_minus[i].params_id());
      evaluator.sub_plain_inplace(enc_x_minus[i], one);
    }
    // case3: num-input - num line
    else if (i > num - input_num)
    {
      vector<double> temps1(slot_count, 0);
      int index = (num - i) * num_batch;
      for (int i = 0; i < slot_count; ++i)
      {
        if (bias_vec[i] == 1 && i >= index)
        {
          // cout <<i<<endl;
          temps1[i] = minus_index;
        }
      }
      // cout <<index/num<<endl;
      // s1[index] = 0;
      PhantomPlaintext one;
      encoder.encode(temps1, enc_x_minus[i].scale(), one);
      evaluator.mod_switch_to_inplace(one, enc_x_minus[i].params_id());
      evaluator.sub_plain_inplace(enc_x_minus[i], one);
    }
    // else{
    //   cout <<"ERROR in computing e^x. "<<endl;
    // }
  }

  // compute e^x_ij
  vector<PhantomCiphertext> exp_x(num);

  vector<double> s1(slot_count, 1);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      s1[i] = 0;
    }
  }

  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, num));

  // —— 准备每线程一个流（拥有型 wrapper） —— //
  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back(); // 默认构造：内部创建并持有一个新 CUDA 流
    }
  }
  if (nthreads == 1)
  {
    stream_pool[0] = *phantom::util::global_variables::default_stream;
  }

  // #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

#pragma omp for schedule(static)
    for (int i = 0; i < num; ++i)
    {
      PhantomCiphertext enc_x_minus_i = deep_copy_cipher(enc_x_minus[i], context, stream);
      exp_x[i] = exp(enc_x_minus_i, context, relin_keys, stream);
      // exp_x[i] = exp(enc_x_minus[i], context, relin_keys, stream);

      // for slot with value 0, minus 1
      // case 0: first line
      if (i == 0)
      {
        PhantomPlaintext one;
        encoder_local.encode(s1, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.sub_plain_inplace(exp_x[i], one, stream);
      }
      // case1: all zero row
      else if (i > input_num && i <= (num - input_num))
      {
        PhantomPlaintext one;
        encoder_local.encode(1, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.sub_plain_inplace(exp_x[i], one, stream);
      }
      // case2: 0 - input_num line
      else if (i <= input_num)
      {
        vector<double> temps1(slot_count, 1);
        int index = num_batch * (input_num - i);
        for (int i = 0; i < slot_count; ++i)
        {
          if (bias_vec[i] == 1 && i < index)
          {
            temps1[i] = 0;
          }
        }
        // cout <<index/num<<endl;
        // s1[index] = 1;
        PhantomPlaintext one;
        encoder_local.encode(temps1, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.sub_plain_inplace(exp_x[i], one, stream);
      }
      // case3: num-input - num line
      else if (i > num - input_num)
      {
        vector<double> temps1(slot_count, 1);
        int index = (num - i) * num_batch;
        for (int i = 0; i < slot_count; ++i)
        {
          if (bias_vec[i] == 1 && i >= index)
          {
            // cout <<i<<endl;
            temps1[i] = 0;
          }
        }
        // cout <<index/num<<endl;
        // s1[index] = 0;
        PhantomPlaintext one;
        encoder_local.encode(temps1, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.sub_plain_inplace(exp_x[i], one, stream);
      }
      // else{
      //   cout <<"ERROR in computing e^x. "<<endl;
      // }
    }
    // bridge_to_default(stream);
    cudaStreamSynchronize(stream.get_stream());
  }

  //  cout <<"    Modulus chain for e^x: "<< seal_context.get_context_data(exp_x[0].parms_id()).chain_depth()<<endl;
  // cout <<log2(exp_x[0].scale())<<endl;
  PhantomPlaintext plain_result;
  vector<double> result;

  /*
    cout <<"TEST result during softmax: "<<endl;

    cout <<"  decrypt of e^x: "<<endl;
    for (int i = 0; i < num; ++i){
      decryptor.decrypt(exp_x[i], plain_result);

      encoder.decode(plain_result, result);
      cout <<i<<"-th: ";
      for (int ind = 0 ; ind < slot_count ; ++ind){
        if(bias_vec[ind] == 1){
          if(result[ind] > 0.0000001){
            cout <<result[ind]<<" ";
          }
          else{
            cout <<"0 ";
          }
        }
      }
    cout <<endl;
    }
   */
  // compute /sum e^x_j
  PhantomCiphertext sum_exp_x = exp_x[0];
  for (int i = 1; i < num; ++i)
  {
    evaluator.add_inplace(sum_exp_x, exp_x[i]);
  }
  /*
    cout <<"  decrypt of sum_exp_(x-8): "<<endl;;
    decryptor.decrypt(sum_exp_x,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<endl;
      }
    }
    cout <<endl;
    */
  /*
    //encode 1/64
    double scalar = 1.0/64.0;
    Plaintext ecd_s;
    encoder.encode(scalar,sum_exp_x.scale(),ecd_s);
    evaluator.mod_switch_to_inplace(ecd_s,sum_exp_x.parms_id());
    evaluator.multiply_plain_inplace(sum_exp_x,ecd_s);
    evaluator.rescale_to_next_inplace(sum_exp_x);
    //cout <<log2(sum_exp_x.scale())<<endl;

    cout <<"  decrypt of 1/64*sum_exp_(x-3): ";
    decryptor.decrypt(sum_exp_x,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
        if(result[ind] > 0.00001){
            cout <<result[ind]<<" ";
          }
          else{
            cout <<"0 ";
          }
      }
    }
    cout <<endl;
  */
  // compute Inv(sum_exp_x)
  PhantomCiphertext inv_sum = inverse(sum_exp_x, context, relin_keys, iter);
  cudaStreamSynchronize(phantom::util::global_variables::default_stream->get_stream());
// cout <<"Modulus chain for inv(sum): "<< seal_context.get_context_data(inv_sum.parms_id()).chain_depth()<<endl;
// cout <<log2(inv_sum.scale())<<endl;
/*
  cout <<"  decrypt of inv(sum_exp_(x-8)): "<<endl;
  decryptor.decrypt(inv_sum,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
       cout <<result[ind]<<endl;
    }
  }
  cout <<endl;
*/
// evaluator.mod_switch_to_inplace(ecd_s,inv_sum.parms_id());
// evaluator.multiply_plain_inplace(inv_sum,ecd_s);
// evaluator.rescale_to_next_inplace(inv_sum);
/*
  vector<double> s0(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] != 0){
      s0[i] = scalar;
    }
  }
  Plaintext ps0;
  encoder.encode(s0,inv_sum.scale(),ps0);
  evaluator.mod_switch_to_inplace(ps0,inv_sum.parms_id());
  evaluator.multiply_plain_inplace(inv_sum,ps0);
  evaluator.rescale_to_next_inplace(inv_sum);
*/
/*
  //cout <<log2(sum_exp_x.scale())<<endl;
  cout <<"  decrypt of 1/64*inv(1/64*sum_exp_(x-3)) = inv(sum_exp_(x-3)): ";
  decryptor.decrypt(inv_sum,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
     // if(result[ind] > 0.00001){
          cout <<result[ind]<<" ";
     //   }
     //   else{
     //     cout <<"0 ";
     //   }
    }
  }
  cout <<endl;
*/
// #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

#pragma omp for schedule(static)
    for (int i = 0; i < num; ++i)
    {
      // evaluator.mod_switch_to_inplace(exp_x[i], inv_sum.params_id());
      // evaluator.multiply(exp_x[i], inv_sum, output[i]);
      // assert(stream.get_stream() == exp_x[i].data_ptr().get_stream());
      evaluator_local.mod_switch_to_inplace(exp_x[i], inv_sum.params_id(), stream);
      evaluator_local.multiply(exp_x[i], inv_sum, output[i], stream);
      evaluator_local.relinearize_inplace(output[i], relin_keys, stream);
      evaluator_local.rescale_to_next_inplace(output[i], stream);
    }
    cudaStreamSynchronize(stream.get_stream());
  }
  /*
    cout <<"  decrypt of e^x/sum_exp_x: ";
    decryptor.decrypt(output[0],plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
        if(result[ind] > 0.00001){
            cout <<result[ind]<<" ";
          }
          else{
            cout <<"0 ";
          }
      }
    }
    cout <<endl;
  */

  return output;
}

vector<PhantomCiphertext> softmax_boot(vector<PhantomCiphertext> &enc_X, vector<int> &bias_vec, int input_num, PhantomContext &context,
                                       PhantomRelinKey &relin_keys, int iter, PhantomSecretKey &sk, Bootstrapper &bootstrapper_att, int layer_id)
{

  int num = enc_X.size();
  double scale = enc_X[0].scale();
  // cout <<"number of ct in output = "<<num<<endl;
  vector<PhantomCiphertext> output(num);

  // CKKSEncoder encoder(context);
  // Evaluator evaluator(context, encoder);
  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  // for test
  Decryptor decryptor(&context, &sk);
  int slot_count = encoder.slot_count();
  // cout <<"slot count = "<<slot_count<<endl;
  int num_batch = slot_count / 128;
  // cout <<"number of batch = "<<num_batch<<endl;
  vector<double> minus_index_vec = {7.5, 9.9, 13.6, 13.3, 9.5, 8, 10.3, 9, 9, 9, 11, 7};

  // compute x_ij - 8
  vector<PhantomCiphertext> enc_x_minus(num);

  double minus_index = minus_index_vec[layer_id];
  //  cout <<"softmax max = "<<minus_index<<endl;
  vector<double> minus(slot_count, 0);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      minus[i] = minus_index;
    }
  }

  // #pragma omp parallel for

  for (int i = 0; i < num; ++i)
  {
    enc_x_minus[i] = enc_X[i];
    // for slot with value neq 0, minus 8
    // case 0: first line
    if (i == 0)
    {
      PhantomPlaintext one;
      encoder.encode(minus, enc_x_minus[i].scale(), one);
      evaluator.mod_switch_to_inplace(one, enc_x_minus[i].params_id());
      evaluator.sub_plain_inplace(enc_x_minus[i], one);
    }
    // case1: all zero row
    else if (i > input_num && i <= (num - input_num))
    {
    }
    // case2: 0 - input_num line
    else if (i <= input_num)
    {
      vector<double> temps1(slot_count, 0);
      int index = num_batch * (input_num - i);
      for (int i = 0; i < slot_count; ++i)
      {
        if (bias_vec[i] == 1 && i < index)
        {
          temps1[i] = minus_index;
        }
      }
      // cout <<index/num<<endl;
      // s1[index] = 1;
      PhantomPlaintext one;
      encoder.encode(temps1, enc_x_minus[i].scale(), one);
      evaluator.mod_switch_to_inplace(one, enc_x_minus[i].params_id());
      evaluator.sub_plain_inplace(enc_x_minus[i], one);
    }
    // case3: num-input - num line
    else if (i > num - input_num)
    {
      vector<double> temps1(slot_count, 0);
      int index = (num - i) * num_batch;
      for (int i = 0; i < slot_count; ++i)
      {
        if (bias_vec[i] == 1 && i >= index)
        {
          // cout <<i<<endl;
          temps1[i] = minus_index;
        }
      }
      // cout <<index/num<<endl;
      // s1[index] = 0;
      PhantomPlaintext one;
      encoder.encode(temps1, enc_x_minus[i].scale(), one);
      evaluator.mod_switch_to_inplace(one, enc_x_minus[i].params_id());
      evaluator.sub_plain_inplace(enc_x_minus[i], one);
    }
    // else{
    //   cout <<"ERROR in computing e^x. "<<endl;
    // }
  }

  // compute e^x_ij
  vector<PhantomCiphertext> exp_x(num);

  vector<double> s1(slot_count, 0);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      s1[i] = 1;
    }
  }

  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, num));

  // —— 准备每线程一个流（拥有型 wrapper） —— //
  if (stream_pool.size() < static_cast<size_t>(nthreads))
  {
    stream_pool.reserve(nthreads);
    for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
    {
      stream_pool.emplace_back(); // 默认构造：内部创建并持有一个新 CUDA 流
    }
  }
  if (nthreads == 1)
  {
    stream_pool[0] = *phantom::util::global_variables::default_stream;
  }

  // #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

#pragma omp for schedule(static)
    for (int i = 0; i < num; ++i)
    {
      PhantomCiphertext enc_x_minus_i = deep_copy_cipher(enc_x_minus[i], context, stream);
      exp_x[i] = exp(enc_x_minus_i, context, relin_keys, stream);

      // for slot with value 0, times 0
      // case 0: first line
      if (i == 0)
      {
        PhantomPlaintext one;
        encoder_local.encode(s1, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.multiply_plain_inplace(exp_x[i], one, stream);
        evaluator_local.rescale_to_next_inplace(exp_x[i], stream);
      }
      // case1: all zero row
      else if (i > input_num && i <= (num - input_num))
      {
        PhantomPlaintext one;
        encoder_local.encode(0, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.multiply_plain_inplace(exp_x[i], one, stream);
        evaluator_local.rescale_to_next_inplace(exp_x[i], stream);
      }
      // case2: 0 - input_num line
      else if (i <= input_num)
      {
        vector<double> temps1(slot_count, 0);
        int index = num_batch * (input_num - i);
        for (int i = 0; i < slot_count; ++i)
        {
          if (bias_vec[i] == 1 && i < index)
          {
            temps1[i] = 1;
          }
        }
        // cout <<index/num<<endl;
        // s1[index] = 1;
        PhantomPlaintext one;
        encoder_local.encode(temps1, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.multiply_plain_inplace(exp_x[i], one, stream);
        evaluator_local.rescale_to_next_inplace(exp_x[i], stream);
      }
      // case3: num-input - num line
      else if (i > num - input_num)
      {
        vector<double> temps1(slot_count, 0);
        int index = (num - i) * num_batch;
        for (int i = 0; i < slot_count; ++i)
        {
          if (bias_vec[i] == 1 && i >= index)
          {
            // cout <<i<<endl;
            temps1[i] = 1;
          }
        }
        // cout <<index/num<<endl;
        // s1[index] = 0;
        PhantomPlaintext one;
        encoder_local.encode(temps1, exp_x[i].scale(), one, stream);
        bridge_to_default(stream);
        evaluator_local.mod_switch_to_inplace(one, exp_x[i].params_id(), stream);
        evaluator_local.multiply_plain_inplace(exp_x[i], one, stream);
        evaluator_local.rescale_to_next_inplace(exp_x[i], stream);
      }
      // else{
      //   cout <<"ERROR in computing e^x. "<<endl;
      // }
      exp_x[i].scale() = scale;
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  /*
  //cout <<"    Modulus chain for e^x: "<< seal_context.get_context_data(exp_x[0].parms_id()).chain_depth()<<endl;
  //cout <<"    Modulus chain for e^x should >= 3"<<endl;
  //cout <<log2(exp_x[0].scale())<<endl;
  Plaintext plain_result;
  vector<double> result;

  cout <<"Decrypt + decode result of e^(x-13): "<<endl;
    //decrypt and decode
    for (int i = 0; i < 5; ++i){
        Plaintext plain_result;
        decryptor.decrypt(exp_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(bias_vec[ind] == 1){
                cout <<result[ind]<<" ";
            }
        }
        cout <<endl;

    }

    for (int i = exp_x.size()-5; i < exp_x.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(exp_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(bias_vec[ind] == 1){
                cout <<result[ind]<<" ";
            }
        }
        cout <<endl;

    }
*/
  // compute /sum e^x_j
  PhantomCiphertext sum_exp_x = exp_x[0];
  for (int i = 1; i < num; ++i)
  {
    evaluator.add_inplace(sum_exp_x, exp_x[i]);
  }
  // evaluator.rescale_to_next_inplace(sum_exp_x);

  // add 1*10^-5
  PhantomPlaintext eps;
  encoder.encode(0.00001, sum_exp_x.params_id(), sum_exp_x.scale(), eps);
  evaluator.add_plain_inplace(sum_exp_x, eps);
  sum_exp_x.scale() = scale;
  /*
    cout <<"  decrypt of sum_exp_(x-13): "<<endl;;
    decryptor.decrypt(sum_exp_x,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<"("<<result[ind]<<" "<<1/result[ind]<<") ";
      }
    }
    cout <<endl;
  */
  // mod switch to the lowest level
  while (context.get_context_data(sum_exp_x.params_id()).chain_depth() != 0)
  {
    evaluator.mod_switch_to_next_inplace(sum_exp_x);
  }
  // cout <<"    Modulus chain before bootstrapping: "<< seal_context.get_context_data(sum_exp_x.parms_id()).chain_depth()<<endl;

  // bootstrapping sum_exp_(x-8)
  PhantomCiphertext rtn;
  bootstrapper_att.bootstrap_3(rtn, sum_exp_x);
  // cout <<"    Modulus chain after bootstrapping: "<< seal_context.get_context_data(rtn.parms_id()).chain_depth()<<endl;
  while (context.get_context_data(rtn.params_id()).chain_depth() > iter + 1 + 3)
  {
    evaluator.mod_switch_to_next_inplace(rtn);
  }
  // cout <<"    Modulus chain after bootstrapping: "<< seal_context.get_context_data(rtn.parms_id()).chain_depth()<<endl;
  // cout <<"    Modulus chain for bootstrapped ct should >= "<<iter+1<<" + modulus chain for e^x"<<endl;
  // compute Inv(sum_exp_x)
  PhantomCiphertext inv_sum = inverse(rtn, context, relin_keys, iter);
  inv_sum.scale() = scale;
  // cout <<"Modulus chain for inv(sum): "<< seal_context.get_context_data(inv_sum.parms_id()).chain_depth()<<endl;
  // cout <<log2(inv_sum.scale())<<endl;
  if (context.get_context_data(exp_x[0].params_id()).chain_depth() < context.get_context_data(inv_sum.params_id()).chain_depth())
  {
    evaluator.mod_switch_to_inplace(inv_sum, exp_x[0].params_id());
  }
  // cout <<"Modulus chain for modswitch(inv(sum)): "<< seal_context.get_context_data(inv_sum.parms_id()).chain_depth()<<endl;
  // cout <<"Modulus chain for modswitch(exp_x): "<< seal_context.get_context_data(exp_x[0].parms_id()).chain_depth()<<endl;
  /*
    cout <<"  decrypt of inv(sum_exp_(x-8)): "<<endl;
    decryptor.decrypt(inv_sum,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
         cout <<result[ind]<<" ";
      }
    }
    cout <<endl;
  */
  // #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

#pragma omp for schedule(static)
    for (int i = 0; i < num; ++i)
    {
      if (context.get_context_data(exp_x[i].params_id()).chain_depth() > context.get_context_data(inv_sum.params_id()).chain_depth())
      {
        evaluator_local.mod_switch_to_inplace(exp_x[i], inv_sum.params_id(), stream);
      }
      evaluator_local.multiply(exp_x[i], inv_sum, output[i], stream);
      evaluator_local.relinearize_inplace(output[i], relin_keys, stream);
      evaluator_local.rescale_to_next_inplace(output[i], stream);
      output[i].scale() = scale;
    }
    cudaStreamSynchronize(stream.get_stream());
  }
  return output;
}

// PhantomCiphertext exp(PhantomCiphertext &x, PhantomContext &context, PhantomRelinKey &relin_keys)
// {
//   // CKKSEncoder encoder(context);
//   // Evaluator evaluator(context, encoder);
//   PhantomCKKSEncoder phantom_encoder(context);
//   // repack the phantom encoder to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   PhantomPlaintext inverse_128;
//   encoder.encode(0.0078125, x.params_id(), x.scale(), inverse_128);
//   // evaluator.mod_switch_to_inplace(inverse_128,x.parms_id());
//   // cout <<"encode 0.0078125"<<endl;

//   PhantomCiphertext output;
//   evaluator.multiply_plain(x, inverse_128, output);
//   evaluator.rescale_to_next_inplace(output);
//   // cout <<"x*0.0078125"<<endl;
//   // cout <<log2(output.scale())<<endl;

//   PhantomPlaintext one;
//   encoder.encode(1.0, output.params_id(), output.scale(), one);
//   // cout <<"encode 1"<<endl;
//   // PhantomCiphertext res;
//   evaluator.add_plain_inplace(output, one);
//   // cout <<"x*0.0078125+1"<<endl;
//   // evaluator.rescale_to_next_inplace(output);
//   // cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output.parms_id()).chain_depth()<<endl;

//   // compute output^128
//   for (int i = 0; i < log2(128); ++i)
//   {
//     // cout <<i<<endl;
//     evaluator.square_inplace(output);
//     evaluator.relinearize_inplace(output, relin_keys);
//     evaluator.rescale_to_next_inplace(output);
//   }
//   // cout <<"(x*0.0078125+1)^128"<<endl;
//   // cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output.parms_id()).chain_depth()<<endl;

//   return output;
// }

// PhantomCiphertext inverse(PhantomCiphertext &x, PhantomContext &context,
//                           PhantomRelinKey &relin_keys, int iter)
// {
//   // by default, iter = 4 (from Nexus)
//   //  CKKSEncoder encoder(seal_context);
//   //  Evaluator evaluator(seal_context, encoder);
//   PhantomCKKSEncoder phantom_encoder(context);
//   // repack the phantom encoder to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   PhantomPlaintext one;
//   encoder.encode(1.0, x.params_id(), x.scale(), one);

//   PhantomCiphertext y;
//   evaluator.sub_plain(x, one, y);
//   evaluator.negate_inplace(y);

//   PhantomCiphertext tmp;
//   evaluator.add_plain(y, one, tmp);

//   PhantomCiphertext res = tmp;
//   for (int i = 0; i < iter; ++i)
//   {
//     evaluator.square_inplace(y);
//     evaluator.relinearize_inplace(y, relin_keys);
//     evaluator.rescale_to_next_inplace(y);

//     // cout <<"y scale = "<<log2(y.scale())<<" , one scale = "<<log2(one.scale())<<endl;
//     encoder.encode(1.0, y.params_id(), y.scale(), one);
//     evaluator.add_plain(y, one, tmp);

//     evaluator.mod_switch_to_inplace(res, tmp.params_id());
//     evaluator.multiply_inplace(res, tmp);
//     evaluator.relinearize_inplace(res, relin_keys);
//     evaluator.rescale_to_next_inplace(res);
//   }

//   return res;
// }

// vector<PhantomCiphertext> softmax(vector<PhantomCiphertext> & enc_X, vector<int> & bias_vec, int input_num, PhantomContext& context,
//   PhantomRelinKey &relin_keys, int iter, PhantomSecretKey & sk){

//   int num = enc_X.size();
//   //cout <<"number of ct in output = "<<num<<endl;
//   vector<PhantomCiphertext> output(num);

//   // PhantomCKKSEncoder encoder(context);
//   // Evaluator evaluator(context, encoder);
//   PhantomCKKSEncoder phantom_encoder(context);
//   // repack the phantom encoder to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   size_t slot_count = encoder.slot_count();
//   //for test
//   Decryptor decryptor(&context, &sk);
//   // int slot_count = encoder.slot_count();
//   //cout <<"slot count = "<<slot_count<<endl;
//   size_t num_batch = slot_count/128;
//   //cout <<"number of batch = "<<num_batch<<endl;

//   //compute x_ij - 8
//   vector<PhantomCiphertext> enc_x_minus(num);

//   double minus_index = 8.1;
//   vector<double> minus(slot_count,0);
//   for (int i = 0; i < slot_count; ++i){
//     if(bias_vec[i] == 1){
//       minus[i] = minus_index;
//     }
//   }

//   for (int i = 0; i < num; ++i){
//     enc_x_minus[i] = enc_X[i];
//     //for slot with value neq 0, minus 8
//     //case 0: first line
//     if(i == 0){
//       PhantomPlaintext one;
//       encoder.encode(minus,enc_x_minus[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,enc_x_minus[i].params_id());
//       evaluator.sub_plain_inplace(enc_x_minus[i],one);
//     }
//     //case1: all zero row
//     else if(i > input_num && i <= (num-input_num)){

//     }
//     //case2: 0 - input_num line
//     else if(i <= input_num){
//       vector<double>temps1(slot_count,0);
//       int index = num_batch * (input_num-i);
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i < index){
//           temps1[i] = minus_index;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 1;
//       PhantomPlaintext one;
//       encoder.encode(temps1,enc_x_minus[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,enc_x_minus[i].params_id());
//       evaluator.sub_plain_inplace(enc_x_minus[i],one);
//     }
//     //case3: num-input - num line
//     else if(i > num-input_num){
//       vector<double>temps1(slot_count,0);
//       int index = (num-i) * num_batch;
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i >= index){
//           //cout <<i<<endl;
//           temps1[i] = minus_index;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 0;
//       PhantomPlaintext one;
//       encoder.encode(temps1,enc_x_minus[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,enc_x_minus[i].params_id());
//       evaluator.sub_plain_inplace(enc_x_minus[i],one);
//     }
//     //else{
//     //  cout <<"ERROR in computing e^x. "<<endl;
//    // }

//   }

//   //compute e^x_ij
//   vector<PhantomCiphertext> exp_x(num);

//   vector<double> s1(slot_count,1);
//   for (int i = 0; i < slot_count; ++i){
//     if(bias_vec[i] == 1){
//       s1[i] = 0;
//     }
//   }

//   // #pragma omp parallel for

//   for (int i = 0; i < num; ++i){
//     exp_x[i] = exp(enc_x_minus[i],context,relin_keys);

//     //for slot with value 0, minus 1
//     //case 0: first line
//     if(i == 0){
//       PhantomPlaintext one;
//       encoder.encode(s1,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.sub_plain_inplace(exp_x[i],one);
//     }
//     //case1: all zero row
//     else if(i > input_num && i <= (num-input_num)){
//       PhantomPlaintext one;
//       encoder.encode(1,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.sub_plain_inplace(exp_x[i],one);
//     }
//     //case2: 0 - input_num line
//     else if(i <= input_num){
//       vector<double>temps1(slot_count,1);
//       int index = num_batch * (input_num-i);
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i < index){
//           temps1[i] = 0;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 1;
//       PhantomPlaintext one;
//       encoder.encode(temps1,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.sub_plain_inplace(exp_x[i],one);
//     }
//     //case3: num-input - num line
//     else if(i > num-input_num){
//       vector<double>temps1(slot_count,1);
//       int index = (num-i) * num_batch;
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i >= index){
//           //cout <<i<<endl;
//           temps1[i] = 0;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 0;
//       PhantomPlaintext one;
//       encoder.encode(temps1,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.sub_plain_inplace(exp_x[i],one);
//     }
//     //else{
//     //  cout <<"ERROR in computing e^x. "<<endl;
//    // }

//   }
// //  cout <<"    Modulus chain for e^x: "<< seal_context.get_context_data(exp_x[0].parms_id()).chain_depth()<<endl;
//   //cout <<log2(exp_x[0].scale())<<endl;
//   PhantomPlaintext plain_result;
//   vector<double> result;

// /*
//   cout <<"TEST result during softmax: "<<endl;

//   cout <<"  decrypt of e^x: "<<endl;
//   for (int i = 0; i < num; ++i){
//     decryptor.decrypt(exp_x[i], plain_result);

//     encoder.decode(plain_result, result);
//     cout <<i<<"-th: ";
//     for (int ind = 0 ; ind < slot_count ; ++ind){
//       if(bias_vec[ind] == 1){
//         if(result[ind] > 0.0000001){
//           cout <<result[ind]<<" ";
//         }
//         else{
//           cout <<"0 ";
//         }
//       }
//     }
//   cout <<endl;
//   }
//  */

//   //compute /sum e^x_j
//   PhantomCiphertext sum_exp_x = exp_x[0];
//   for (int i = 1; i < num; ++i){
//     evaluator.add_inplace(sum_exp_x,exp_x[i]);
//   }
// /*
//   cout <<"  decrypt of sum_exp_(x-8): "<<endl;;
//   decryptor.decrypt(sum_exp_x,plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//         cout <<result[ind]<<endl;
//     }
//   }
//   cout <<endl;
//   */
// /*
//   //encode 1/64
//   double scalar = 1.0/64.0;
//   Plaintext ecd_s;
//   encoder.encode(scalar,sum_exp_x.scale(),ecd_s);
//   evaluator.mod_switch_to_inplace(ecd_s,sum_exp_x.parms_id());
//   evaluator.multiply_plain_inplace(sum_exp_x,ecd_s);
//   evaluator.rescale_to_next_inplace(sum_exp_x);
//   //cout <<log2(sum_exp_x.scale())<<endl;

//   cout <<"  decrypt of 1/64*sum_exp_(x-3): ";
//   decryptor.decrypt(sum_exp_x,plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//       if(result[ind] > 0.00001){
//           cout <<result[ind]<<" ";
//         }
//         else{
//           cout <<"0 ";
//         }
//     }
//   }
//   cout <<endl;
// */
//   //compute Inv(sum_exp_x)
//   PhantomCiphertext inv_sum = inverse(sum_exp_x,context,relin_keys,iter);
//  // cout <<"Modulus chain for inv(sum): "<< seal_context.get_context_data(inv_sum.parms_id()).chain_depth()<<endl;
//   //cout <<log2(inv_sum.scale())<<endl;
// /*
//   cout <<"  decrypt of inv(sum_exp_(x-8)): "<<endl;
//   decryptor.decrypt(inv_sum,plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//        cout <<result[ind]<<endl;
//     }
//   }
//   cout <<endl;
// */
//   //evaluator.mod_switch_to_inplace(ecd_s,inv_sum.parms_id());
//   //evaluator.multiply_plain_inplace(inv_sum,ecd_s);
//   //evaluator.rescale_to_next_inplace(inv_sum);
// /*
//   vector<double> s0(slot_count,0);
//   for (int i = 0; i < slot_count; ++i){
//     if(bias_vec[i] != 0){
//       s0[i] = scalar;
//     }
//   }
//   Plaintext ps0;
//   encoder.encode(s0,inv_sum.scale(),ps0);
//   evaluator.mod_switch_to_inplace(ps0,inv_sum.parms_id());
//   evaluator.multiply_plain_inplace(inv_sum,ps0);
//   evaluator.rescale_to_next_inplace(inv_sum);
// */
// /*
//   //cout <<log2(sum_exp_x.scale())<<endl;
//   cout <<"  decrypt of 1/64*inv(1/64*sum_exp_(x-3)) = inv(sum_exp_(x-3)): ";
//   decryptor.decrypt(inv_sum,plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//      // if(result[ind] > 0.00001){
//           cout <<result[ind]<<" ";
//      //   }
//      //   else{
//      //     cout <<"0 ";
//      //   }
//     }
//   }
//   cout <<endl;
// */
//   // #pragma omp parallel for

//   for (int i = 0; i < num; ++i){
//     evaluator.mod_switch_to_inplace(exp_x[i],inv_sum.params_id());
//     evaluator.multiply(exp_x[i],inv_sum,output[i]);
//     evaluator.relinearize_inplace(output[i],relin_keys);
//     evaluator.rescale_to_next_inplace(output[i]);
//   }
// /*
//   cout <<"  decrypt of e^x/sum_exp_x: ";
//   decryptor.decrypt(output[0],plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//       if(result[ind] > 0.00001){
//           cout <<result[ind]<<" ";
//         }
//         else{
//           cout <<"0 ";
//         }
//     }
//   }
//   cout <<endl;
// */

//   return output;

// }

// vector<PhantomCiphertext> softmax_boot(vector<PhantomCiphertext> & enc_X, vector<int> & bias_vec, int input_num, PhantomContext& context,
//   PhantomRelinKey &relin_keys, int iter, PhantomSecretKey & sk, Bootstrapper& bootstrapper_att, int layer_id){

//   int num = enc_X.size();
//   double scale = enc_X[0].scale();
//   //cout <<"number of ct in output = "<<num<<endl;
//   vector<PhantomCiphertext> output(num);

//   // CKKSEncoder encoder(context);
//   // Evaluator evaluator(context, encoder);
//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   //for test
//   Decryptor decryptor(&context, &sk);
//   int slot_count = encoder.slot_count();
//   //cout <<"slot count = "<<slot_count<<endl;
//   int num_batch = slot_count/128;
//   //cout <<"number of batch = "<<num_batch<<endl;
//   vector<double> minus_index_vec = {7.5, 9.9, 13.6, 13.3, 9.5, 8, 10.3, 9, 9, 9, 11, 7};

//   //compute x_ij - 8
//   vector<PhantomCiphertext> enc_x_minus(num);

//   double minus_index = minus_index_vec[layer_id];
// //  cout <<"softmax max = "<<minus_index<<endl;
//   vector<double> minus(slot_count,0);
//   for (int i = 0; i < slot_count; ++i){
//     if(bias_vec[i] == 1){
//       minus[i] = minus_index;
//     }
//   }

//   // #pragma omp parallel for

//   for (int i = 0; i < num; ++i){
//     enc_x_minus[i] = enc_X[i];
//     //for slot with value neq 0, minus 8
//     //case 0: first line
//     if(i == 0){
//       PhantomPlaintext one;
//       encoder.encode(minus,enc_x_minus[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,enc_x_minus[i].params_id());
//       evaluator.sub_plain_inplace(enc_x_minus[i],one);
//     }
//     //case1: all zero row
//     else if(i > input_num && i <= (num-input_num)){

//     }
//     //case2: 0 - input_num line
//     else if(i <= input_num){
//       vector<double>temps1(slot_count,0);
//       int index = num_batch * (input_num-i);
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i < index){
//           temps1[i] = minus_index;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 1;
//       PhantomPlaintext one;
//       encoder.encode(temps1,enc_x_minus[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,enc_x_minus[i].params_id());
//       evaluator.sub_plain_inplace(enc_x_minus[i],one);
//     }
//     //case3: num-input - num line
//     else if(i > num-input_num){
//       vector<double>temps1(slot_count,0);
//       int index = (num-i) * num_batch;
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i >= index){
//           //cout <<i<<endl;
//           temps1[i] = minus_index;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 0;
//       PhantomPlaintext one;
//       encoder.encode(temps1,enc_x_minus[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,enc_x_minus[i].params_id());
//       evaluator.sub_plain_inplace(enc_x_minus[i],one);
//     }
//     //else{
//     //  cout <<"ERROR in computing e^x. "<<endl;
//    // }

//   }

//   //compute e^x_ij
//   vector<PhantomCiphertext> exp_x(num);

//   vector<double> s1(slot_count,0);
//   for (int i = 0; i < slot_count; ++i){
//     if(bias_vec[i] == 1){
//       s1[i] = 1;
//     }
//   }

//   // #pragma omp parallel for

//   for (int i = 0; i < num; ++i){
//     exp_x[i] = exp(enc_x_minus[i],context,relin_keys);

//     //for slot with value 0, times 0
//     //case 0: first line
//     if(i == 0){
//       PhantomPlaintext one;
//       encoder.encode(s1,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.multiply_plain_inplace(exp_x[i],one);
//       evaluator.rescale_to_next_inplace(exp_x[i]);
//     }
//     //case1: all zero row
//     else if(i > input_num && i <= (num-input_num)){
//       PhantomPlaintext one;
//       encoder.encode(0,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.multiply_plain_inplace(exp_x[i],one);
//       evaluator.rescale_to_next_inplace(exp_x[i]);
//     }
//     //case2: 0 - input_num line
//     else if(i <= input_num){
//       vector<double>temps1(slot_count,0);
//       int index = num_batch * (input_num-i);
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i < index){
//           temps1[i] = 1;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 1;
//       PhantomPlaintext one;
//       encoder.encode(temps1,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.multiply_plain_inplace(exp_x[i],one);
//       evaluator.rescale_to_next_inplace(exp_x[i]);
//     }
//     //case3: num-input - num line
//     else if(i > num-input_num){
//       vector<double>temps1(slot_count,0);
//       int index = (num-i) * num_batch;
//       for (int i = 0; i < slot_count; ++i){
//         if(bias_vec[i] == 1 && i >= index){
//           //cout <<i<<endl;
//           temps1[i] = 1;
//         }
//       }
//       //cout <<index/num<<endl;
//       //s1[index] = 0;
//       PhantomPlaintext one;
//       encoder.encode(temps1,exp_x[i].scale(),one);
//       evaluator.mod_switch_to_inplace(one,exp_x[i].params_id());
//       evaluator.multiply_plain_inplace(exp_x[i],one);
//       evaluator.rescale_to_next_inplace(exp_x[i]);
//     }
//     //else{
//     //  cout <<"ERROR in computing e^x. "<<endl;
//    // }
//     exp_x[i].scale() = scale;

//   }
//   /*
//   //cout <<"    Modulus chain for e^x: "<< seal_context.get_context_data(exp_x[0].parms_id()).chain_depth()<<endl;
//   //cout <<"    Modulus chain for e^x should >= 3"<<endl;
//   //cout <<log2(exp_x[0].scale())<<endl;
//   Plaintext plain_result;
//   vector<double> result;

//   cout <<"Decrypt + decode result of e^(x-13): "<<endl;
//     //decrypt and decode
//     for (int i = 0; i < 5; ++i){
//         Plaintext plain_result;
//         decryptor.decrypt(exp_x[i], plain_result);
//         vector<double> result;
//         encoder.decode(plain_result, result);
//         cout <<i+1<<"-th ciphertext: ";
//         for (int ind = 0 ; ind < slot_count ; ++ind){
//             if(bias_vec[ind] == 1){
//                 cout <<result[ind]<<" ";
//             }
//         }
//         cout <<endl;

//     }

//     for (int i = exp_x.size()-5; i < exp_x.size(); ++i){
//         Plaintext plain_result;
//         decryptor.decrypt(exp_x[i], plain_result);
//         vector<double> result;
//         encoder.decode(plain_result, result);
//         cout <<i+1<<"-th ciphertext: ";
//         for (int ind = 0 ; ind < slot_count ; ++ind){
//             if(bias_vec[ind] == 1){
//                 cout <<result[ind]<<" ";
//             }
//         }
//         cout <<endl;

//     }
// */
//   //compute /sum e^x_j
//   PhantomCiphertext sum_exp_x = exp_x[0];
//   for (int i = 1; i < num; ++i){
//     evaluator.add_inplace(sum_exp_x,exp_x[i]);
//   }
//   //evaluator.rescale_to_next_inplace(sum_exp_x);

//   //add 1*10^-5
//   PhantomPlaintext eps;
//   encoder.encode(0.00001, sum_exp_x.params_id(), sum_exp_x.scale(), eps);
//   evaluator.add_plain_inplace(sum_exp_x,eps);
//   sum_exp_x.scale()=scale;
// /*
//   cout <<"  decrypt of sum_exp_(x-13): "<<endl;;
//   decryptor.decrypt(sum_exp_x,plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//         cout <<"("<<result[ind]<<" "<<1/result[ind]<<") ";
//     }
//   }
//   cout <<endl;
// */
//   //mod switch to the lowest level
//   while(context.get_context_data(sum_exp_x.params_id()).chain_depth() != 0){
//     evaluator.mod_switch_to_next_inplace(sum_exp_x);
//   }
//   //cout <<"    Modulus chain before bootstrapping: "<< seal_context.get_context_data(sum_exp_x.parms_id()).chain_depth()<<endl;

//   //bootstrapping sum_exp_(x-8)
//   PhantomCiphertext rtn;
//   bootstrapper_att.bootstrap_3(rtn,sum_exp_x);
//   //cout <<"    Modulus chain after bootstrapping: "<< seal_context.get_context_data(rtn.parms_id()).chain_depth()<<endl;
//   while (context.get_context_data(rtn.params_id()).chain_depth() > iter + 1 + 3){
//     evaluator.mod_switch_to_next_inplace(rtn);
//   }
//   //cout <<"    Modulus chain after bootstrapping: "<< seal_context.get_context_data(rtn.parms_id()).chain_depth()<<endl;
//   //cout <<"    Modulus chain for bootstrapped ct should >= "<<iter+1<<" + modulus chain for e^x"<<endl;
//   //compute Inv(sum_exp_x)
//   PhantomCiphertext inv_sum = inverse(rtn,context,relin_keys,iter);
//   inv_sum.scale() = scale;
//   //cout <<"Modulus chain for inv(sum): "<< seal_context.get_context_data(inv_sum.parms_id()).chain_depth()<<endl;
//   //cout <<log2(inv_sum.scale())<<endl;
//   if(context.get_context_data(exp_x[0].params_id()).chain_depth()
//       <context.get_context_data(inv_sum.params_id()).chain_depth()){
//       evaluator.mod_switch_to_inplace(inv_sum,exp_x[0].params_id());
//     }
//   //cout <<"Modulus chain for modswitch(inv(sum)): "<< seal_context.get_context_data(inv_sum.parms_id()).chain_depth()<<endl;
//   //cout <<"Modulus chain for modswitch(exp_x): "<< seal_context.get_context_data(exp_x[0].parms_id()).chain_depth()<<endl;
// /*
//   cout <<"  decrypt of inv(sum_exp_(x-8)): "<<endl;
//   decryptor.decrypt(inv_sum,plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//        cout <<result[ind]<<" ";
//     }
//   }
//   cout <<endl;
// */
//   // #pragma omp parallel for

//   for (int i = 0; i < num; ++i){
//     if(context.get_context_data(exp_x[i].params_id()).chain_depth()
//       >context.get_context_data(inv_sum.params_id()).chain_depth()){
//       evaluator.mod_switch_to_inplace(exp_x[i],inv_sum.params_id());
//     }
//     evaluator.multiply(exp_x[i],inv_sum,output[i]);
//     evaluator.relinearize_inplace(output[i],relin_keys);
//     evaluator.rescale_to_next_inplace(output[i]);
//     output[i].scale() = scale;
//   }

//   return output;

// }