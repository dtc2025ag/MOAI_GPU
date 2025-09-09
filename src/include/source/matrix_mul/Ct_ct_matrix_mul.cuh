#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

vector<PhantomCiphertext> ct_ct_matrix_mul_colpacking(vector<PhantomCiphertext> &enc_X,
                                                      vector<PhantomCiphertext> &enc_W, PhantomGaloisKey &RotK, PhantomRelinKey &relin_keys,
                                                      PhantomContext &context, int col_X, int row_X, int col_W, int row_W, int num_batch)
{

  // vector<PhantomCiphertext> output(row_X);
  vector<PhantomCiphertext> output(static_cast<size_t>(row_X));
  double scale = enc_X[0].scale();

  if (col_X != col_W || row_X != row_W)
  {
    cout << "ERROR: bad dimensions of X or W. " << endl;
    return output;
  }

  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, row_X));

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

  // cudaEvent_t ev_start, ev_stop;
  // cudaEventCreate(&ev_start);
  // cudaEventCreate(&ev_stop);

  // 可选：单独的计时流，避免用默认流 0
  // cudaStream_t timing_stream = nullptr;
  // cudaStreamCreateWithFlags(&timing_stream, cudaStreamNonBlocking);

  // PhantomCKKSEncoder phantom_encoder(context);
  // // pack Phantom to SEAL style
  // Encoder encoder(&context, &phantom_encoder);
  // Evaluator evaluator(&context, &phantom_encoder);

  // #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

    // std::vector<PhantomCiphertext> X_local(static_cast<size_t>(col_X));
    // for (int j = 0; j < col_X; ++j)
    // {
    //   // 若 multiply_plain / add_inplace 不会修改输入，可直接引用 enc_X[j] 而无需拷贝
    //   X_local[static_cast<size_t>(j)] = deep_copy_cipher(enc_X[j], context, stream);
    // }

    // cudaStreamWaitEvent(stream.get_stream(), ev_start, 0);
    // cudaDeviceSynchronize();
// 确保所有线程都已经设置好 wait 之后再开枪
// #pragma omp barrier
// #pragma omp single
//     {
//       // 起跑枪：现在才开始计时，预处理不包含
//       cudaEventRecord(ev_start, timing_stream ? timing_stream : 0);
//     }
#pragma omp for schedule(static)
    for (int i = 0; i < row_X; ++i)
    {
      // cout << "Processing row " << i << " in thread " << tid << endl;
      PhantomCiphertext acc, temp;
      vector<PhantomCiphertext> copy_w(col_X);
      // for (int i = 0; i < col_X; ++i)
      // {
      //   copy_w[i] = deep_copy_cipher(enc_W[i], context, stream);
      // }
      for (int j = 0; j < col_X; ++j)
      {
        // copy_w[j] = enc_W[j];
        copy_w[j] = deep_copy_cipher(enc_W[j], context, stream);
        if (i > 0)
        {
          evaluator_local.rotate_vector_inplace(copy_w[j], i * num_batch, RotK, stream);
          // cudaStreamSynchronize(stream.get_stream()); // 当前仍需要
          // PhantomCiphertext rotated;
          // evaluator_local.rotate_vector(enc_W[j], i * num_batch, RotK, rotated); // 输出到 rotated
          // copy_w[j] = std::move(rotated);
        }
      }

      // evaluator_local.multiply(X_local[0], copy_w[0], acc, stream);
      PhantomCiphertext x0 = deep_copy_cipher(enc_X[0], context, stream);
      evaluator_local.multiply(x0, copy_w[0], acc, stream);
      // evaluator_local.multiply(enc_X[0], copy_w[0], acc, stream);
      // evaluator.relinearize_inplace(output[i],relin_keys);
      // evaluator.rescale_to_next_inplace(output[i]);

      for (int j = 1; j < col_X; ++j)
      {
        // PhantomCiphertext temp;
        // evaluator_local.multiply(X_local[j], copy_w[j], temp, stream);
        PhantomCiphertext xj = deep_copy_cipher(enc_X[j], context, stream);
        evaluator_local.multiply(xj, copy_w[j], temp, stream);
        // evaluator_local.multiply(enc_X[j], copy_w[j], temp, stream);
        // evaluator.relinearize_inplace(temp,relin_keys);
        // evaluator.rescale_to_next_inplace(temp);
        evaluator_local.add_inplace(acc, temp, stream);
      }

      // put the relinearization and rescale to the end of sum.
      evaluator_local.relinearize_inplace(acc, relin_keys, stream);
      evaluator_local.rescale_to_next_inplace(acc, stream);
      output[static_cast<size_t>(i)] = std::move(acc);
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // 在并行区外创建/记录 ev_done 更清晰：每个线程结束前在各自流上 Record
  std::vector<cudaEvent_t> ev_done(nthreads);
  for (int i = 0; i < nthreads; ++i)
  {
    cudaEventCreateWithFlags(&ev_done[i], cudaEventDisableTiming);
    cudaEventRecord(ev_done[i], stream_pool[i].get_stream());
  }

  // 聚合所有 done 到计时流
  // for (int i = 0; i < nthreads; ++i)
  // {
  //   cudaStreamWaitEvent(timing_stream ? timing_stream : 0, ev_done[i], 0);
  // }

  // 记录 stop 并计算时间
  // cudaEventRecord(ev_stop, timing_stream ? timing_stream : 0);
  // cudaEventSynchronize(ev_stop);

  // float ms = 0.f;
  // cudaEventElapsedTime(&ms, ev_start, ev_stop);
  // cout << "Ct-Pt compute time = " << ms << " ms\n";

  // 清理
  // for (auto &e : ev_done)
  //   cudaEventDestroy(e);
  // cudaEventDestroy(ev_start);
  // cudaEventDestroy(ev_stop);
  // if (timing_stream)
  //   cudaStreamDestroy(timing_stream);
  // cudaDeviceSynchronize();
  return output;
}

vector<PhantomCiphertext> ct_ct_matrix_mul_diagpacking(vector<PhantomCiphertext> &enc_X,
                                                       vector<PhantomCiphertext> &enc_W, PhantomGaloisKey &RotK, PhantomRelinKey &relin_keys,
                                                       PhantomContext &context, int col_X, int row_X, int col_W, int row_W, int num_batch)
{

  // X: diag encoding
  // W: column encoding
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, 32));

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

  // PhantomCKKSEncoder phantom_encoder(context);
  // Encoder encoder(&context, &phantom_encoder);
  // Evaluator evaluator(&context, &phantom_encoder);
  double scale = enc_X[0].scale();

  vector<PhantomCiphertext> output(col_W);

  int g = sqrt((double)col_X);
  if (g * g < col_X)
  {
    g++;
  }

  int b = col_X / g;
  if (b * g < col_X)
  {
    b++;
  }

  // rotate X

  vector<PhantomCiphertext> rot_enc_X(row_X);

  // cudaEvent_t ev_start_rot, ev_stop_rot;
  // cudaEventCreate(&ev_start_rot);
  // cudaEventCreate(&ev_stop_rot);

  // // 可选：单独的计时流，避免用默认流 0
  // cudaStream_t timing_stream_rot = nullptr;
  // cudaStreamCreateWithFlags(&timing_stream_rot, cudaStreamNonBlocking);

  // 确保所有线程都已经设置好 wait 之后再开枪
// #pragma omp barrier
// #pragma omp single
//   {
//     // 起跑枪：现在才开始计时，预处理不包含
//     cudaEventRecord(ev_start_rot, timing_stream_rot ? timing_stream_rot : 0);
//   }

  // #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {

    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

    // cudaStreamWaitEvent(stream.get_stream(), ev_start_rot, 0);

#pragma omp for schedule(static)
    for (int i = 0; i < b; ++i)
    {
      for (int j = 0; j < g; ++j)
      {
        int index = i * g + j;
        if (index >= row_X)
        {
          break;
        }
        else
        {
          int rot_ind = (col_X - i * g) * num_batch;
          if (rot_ind != col_X * num_batch)
          {
            PhantomCiphertext xindex = deep_copy_cipher(enc_X[index], context, stream);
            evaluator_local.rotate_vector(xindex, rot_ind, RotK, rot_enc_X[index], stream);
            // evaluator_local.rotate_vector(enc_X[index], rot_ind, RotK, rot_enc_X[index], stream);
          }
          else
          {
            rot_enc_X[index] = deep_copy_cipher(enc_X[index], context, stream);
          }
        }
      }
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // 在并行区外创建/记录 ev_done 更清晰：每个线程结束前在各自流上 Record
  // std::vector<cudaEvent_t> ev_done_rot(nthreads);
  // for (int i = 0; i < nthreads; ++i)
  // {
  //   cudaEventCreateWithFlags(&ev_done_rot[i], cudaEventDisableTiming);
  //   cudaEventRecord(ev_done_rot[i], stream_pool[i].get_stream());
  // }

  // 聚合所有 done 到计时流
  // for (int i = 0; i < nthreads; ++i)
  // {
  //   cudaStreamWaitEvent(timing_stream_rot ? timing_stream_rot : 0, ev_done_rot[i], 0);
  // }

  // 记录 stop 并计算时间
  // cudaEventRecord(ev_stop_rot, timing_stream_rot ? timing_stream_rot : 0);
  // cudaEventSynchronize(ev_stop_rot);

  // float ms = 0.f;
  // cudaEventElapsedTime(&ms, ev_start_rot, ev_stop_rot);
  // cout << "Ct-ct diag rotate compute time = " << ms << " ms\n";

  // 清理
  // for (auto &e : ev_done_rot)
  //   cudaEventDestroy(e);
  // cudaEventDestroy(ev_start_rot);
  // cudaEventDestroy(ev_stop_rot);
  // if (timing_stream_rot)
  //   cudaStreamDestroy(timing_stream_rot);

  // cudaEvent_t ev_start_bsgs, ev_stop_bsgs;
  // cudaEventCreate(&ev_start_bsgs);
  // cudaEventCreate(&ev_stop_bsgs);

  // // 可选：单独的计时流，避免用默认流 0
  // cudaStream_t timing_stream_bsgs = nullptr;
  // cudaStreamCreateWithFlags(&timing_stream_bsgs, cudaStreamNonBlocking);

  // baby step + gaint step (col_w times)
  //  #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
  {
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

    // cudaStreamWaitEvent(stream.get_stream(), ev_start_bsgs, 0);

    // 确保所有线程都已经设置好 wait 之后再开枪
// #pragma omp barrier
// #pragma omp single
//     {
//       // 起跑枪：现在才开始计时，预处理不包含
//       cudaEventRecord(ev_start_bsgs, timing_stream_bsgs ? timing_stream_bsgs : 0);
//     }

#pragma omp for schedule(static)
    for (int i = 0; i < col_W; ++i)
    {
      // baby step
      // vector<PhantomCiphertext> c_g(g, enc_W[i]);
      vector<PhantomCiphertext> c_g(static_cast<size_t>(g));
      for (int j = 0; j < g; ++j)
      {
        c_g[j] = deep_copy_cipher(enc_W[i], context, stream);
      }

      for (int j = 1; j < g; ++j)
      {
        evaluator_local.rotate_vector_inplace(c_g[j], j * num_batch, RotK, stream);
      }
      // cout <<"baby step. "<<endl;

      // gaint step
      vector<PhantomCiphertext> out(b);
      for (int j = 0; j < b; ++j)
      {
        // cout <<"j = "<<j<<endl;
        for (int k = 0; k < g; ++k)
        {
          int index = j * g + k;
          if (index >= col_X)
          {
            break;
          }
          // PhantomCiphertext c_gk = deep_copy_cipher(c_g[k], context, stream);
          if (k == 0)
          {

            // evaluator_local.multiply(c_gk, rot_enc_X[index], out[j], stream);
            evaluator_local.multiply(c_g[k], rot_enc_X[index], out[j], stream);
            // evaluator.relinearize_inplace(out[j],relin_keys);
            // evaluator.rescale_to_next_inplace(out[j]);
          }
          else
          {
            PhantomCiphertext temp;
            evaluator_local.multiply(c_g[k], rot_enc_X[index], temp, stream);
            // evaluator.relinearize_inplace(temp,relin_keys);
            // evaluator.rescale_to_next_inplace(temp);
            evaluator_local.add_inplace(out[j], temp, stream);
          }
        }
        evaluator_local.relinearize_inplace(out[j], relin_keys, stream);
        evaluator_local.rescale_to_next_inplace(out[j], stream);
        out[j].scale() = scale;
      }
      for (int j = 0; j < b; ++j)
      {
        if (j == 0)
        {
          output[i] = out[j];
        }
        else
        {
          evaluator_local.rotate_vector_inplace(out[j], j * g * num_batch, RotK, stream);
          evaluator_local.add_inplace(output[i], out[j], stream);
        }
      }
    }
    cudaStreamSynchronize(stream.get_stream());
  }

  // 在并行区外创建/记录 ev_done 更清晰：每个线程结束前在各自流上 Record
  // std::vector<cudaEvent_t> ev_done_bsgs(nthreads);
  // for (int i = 0; i < nthreads; ++i)
  // {
  //   cudaEventCreateWithFlags(&ev_done_bsgs[i], cudaEventDisableTiming);
  //   cudaEventRecord(ev_done_bsgs[i], stream_pool[i].get_stream());
  // }

  // 聚合所有 done 到计时流
  // for (int i = 0; i < nthreads; ++i)
  // {
  //   cudaStreamWaitEvent(timing_stream_bsgs ? timing_stream_bsgs : 0, ev_done_bsgs[i], 0);
  // }

  // // 记录 stop 并计算时间
  // cudaEventRecord(ev_stop_bsgs, timing_stream_bsgs ? timing_stream_bsgs : 0);
  // cudaEventSynchronize(ev_stop_bsgs);

  // // float ms = 0.f;
  // cudaEventElapsedTime(&ms, ev_start_bsgs, ev_stop_bsgs);
  // cout << "Ct-ct diag bsgs compute time = " << ms << " ms\n";

  // // 清理
  // for (auto &e : ev_done_bsgs)
  //   cudaEventDestroy(e);
  // cudaEventDestroy(ev_start_bsgs);
  // cudaEventDestroy(ev_stop_bsgs);
  // if (timing_stream_bsgs)
  //   cudaStreamDestroy(timing_stream_bsgs);

  return output;
}

// vector<PhantomCiphertext> ct_ct_matrix_mul_colpacking(vector<PhantomCiphertext> &enc_X,
//                                                       vector<PhantomCiphertext> &enc_W, PhantomGaloisKey &RotK, PhantomRelinKey &relin_keys,
//                                                       PhantomContext &context, int col_X, int row_X, int col_W, int row_W, int num_batch)
// {

//   vector<PhantomCiphertext> output(row_X);
//   double scale = enc_X[0].scale();

//   if (col_X != col_W || row_X != row_W)
//   {
//     cout << "ERROR: bad dimensions of X or W. " << endl;
//     return output;
//   }

//   PhantomCKKSEncoder phantom_encoder(context);
//   // pack Phantom to SEAL style
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   // #pragma omp parallel for

//   for (int i = 0; i < row_X; ++i)
//   {
//     vector<PhantomCiphertext> copy_w(col_X);
//     for (int j = 0; j < col_X; ++j)
//     {
//       copy_w[j] = enc_W[j];
//       if (i > 0)
//       {
//         evaluator.rotate_vector_inplace(copy_w[j], i * num_batch, RotK);
//       }
//     }
//     evaluator.multiply(enc_X[0], copy_w[0], output[i]);
//     // evaluator.relinearize_inplace(output[i],relin_keys);
//     // evaluator.rescale_to_next_inplace(output[i]);

//     for (int j = 1; j < col_X; ++j)
//     {
//       PhantomCiphertext temp;
//       evaluator.multiply(enc_X[j], copy_w[j], temp);
//       // evaluator.relinearize_inplace(temp,relin_keys);
//       // evaluator.rescale_to_next_inplace(temp);
//       evaluator.add_inplace(output[i], temp);
//     }

//     // put the relinearization and rescale to the end of sum.
//     evaluator.relinearize_inplace(output[i], relin_keys);
//     evaluator.rescale_to_next_inplace(output[i]);
//     output[i].scale() = scale;
//   }

//   return output;
// }

// vector<PhantomCiphertext> ct_ct_matrix_mul_diagpacking(vector<PhantomCiphertext> &enc_X,
//                                                        vector<PhantomCiphertext> &enc_W, PhantomGaloisKey &RotK, PhantomRelinKey &relin_keys,
//                                                        PhantomContext &context, int col_X, int row_X, int col_W, int row_W, int num_batch)
// {

//   // X: diag encoding
//   // W: column encoding

//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   double scale = enc_X[0].scale();

//   vector<PhantomCiphertext> output(col_W);

//   int g = sqrt((double)col_X);
//   if (g * g < col_X)
//   {
//     g++;
//   }

//   int b = col_X / g;
//   if (b * g < col_X)
//   {
//     b++;
//   }

//   // rotate X

//   vector<PhantomCiphertext> rot_enc_X(row_X);

//   // #pragma omp parallel for

//   for (int i = 0; i < b; ++i)
//   {
//     for (int j = 0; j < g; ++j)
//     {
//       int index = i * g + j;
//       if (index >= row_X)
//       {
//         break;
//       }
//       else
//       {
//         int rot_ind = (col_X - i * g) * num_batch;
//         if (rot_ind != col_X * num_batch)
//         {
//           evaluator.rotate_vector(enc_X[index], rot_ind, RotK, rot_enc_X[index]);
//         }
//         else
//         {
//           rot_enc_X[index] = enc_X[index];
//         }
//       }
//     }
//   }

//   // baby step + gaint step (col_w times)
//   //  #pragma omp parallel for

//   for (int i = 0; i < col_W; ++i)
//   {
//     // baby step
//     vector<PhantomCiphertext> c_g(g, enc_W[i]);

//     for (int j = 1; j < g; ++j)
//     {
//       evaluator.rotate_vector_inplace(c_g[j], j * num_batch, RotK);
//     }
//     // cout <<"baby step. "<<endl;

//     // gaint step
//     vector<PhantomCiphertext> out(b);
//     for (int j = 0; j < b; ++j)
//     {
//       // cout <<"j = "<<j<<endl;
//       for (int k = 0; k < g; ++k)
//       {
//         int index = j * g + k;
//         if (index >= col_X)
//         {
//           break;
//         }
//         if (k == 0)
//         {
//           evaluator.multiply(c_g[k], rot_enc_X[index], out[j]);
//           // evaluator.relinearize_inplace(out[j],relin_keys);
//           // evaluator.rescale_to_next_inplace(out[j]);
//         }
//         else
//         {
//           PhantomCiphertext temp;
//           evaluator.multiply(c_g[k], rot_enc_X[index], temp);
//           // evaluator.relinearize_inplace(temp,relin_keys);
//           // evaluator.rescale_to_next_inplace(temp);
//           evaluator.add_inplace(out[j], temp);
//         }
//       }
//       evaluator.relinearize_inplace(out[j], relin_keys);
//       evaluator.rescale_to_next_inplace(out[j]);
//       out[j].scale() = scale;
//     }
//     for (int j = 0; j < b; ++j)
//     {
//       if (j == 0)
//       {
//         output[i] = out[j];
//       }
//       else
//       {
//         evaluator.rotate_vector_inplace(out[j], j * g * num_batch, RotK);
//         evaluator.add_inplace(output[i], out[j]);
//       }
//     }
//   }

//   return output;
// }