#pragma once
#include <random>

#include "include.cuh"
// #include "bootstrapping/Bootstrapper.cuh"
#include "phantom.h"

using namespace std;
using namespace phantom;

void random_real(vector<double> &vec, size_t size)
{
  random_device rn;
  mt19937_64 rnd(rn());
  thread_local std::uniform_real_distribution<double> distribution(-1, 1);

  vec.reserve(size);

  for (size_t i = 0; i < size; i++)
  {
    vec[i] = distribution(rnd);
  }
}

void bootstrapping_test()
{
  long boundary_K = 25;
  long deg = 59;
  long scale_factor = 2;
  long inverse_deg = 1;

  // The following parameters have been adjusted to satisfy the memory constraints of an A100 GPU
  long logN = 15; // 16 -> 15
  long loge = 10;

  long logn = 13; // 14 -> 13
  long sparse_slots = (1 << logn);

  int logp = 46;
  int logq = 51;
  int log_special_prime = 51;

  int secret_key_hamming_weight = 192;

  int remaining_level = 16;
  int boot_level = 14; // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
  int total_level = remaining_level + boot_level;

  vector<int> coeff_bit_vec;
  coeff_bit_vec.push_back(logq);
  for (int i = 0; i < remaining_level; i++)
  {
    coeff_bit_vec.push_back(logp);
  }
  for (int i = 0; i < boot_level; i++)
  {
    coeff_bit_vec.push_back(logq);
  }
  coeff_bit_vec.push_back(log_special_prime);

  std::cout << "Setting Parameters..." << endl;
  phantom::EncryptionParameters parms(scheme_type::ckks);
  size_t poly_modulus_degree = (size_t)(1 << logN);
  double scale = pow(2.0, logp);

  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
  parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
  parms.set_sparse_slots(sparse_slots);

  PhantomContext context(parms);

  PhantomSecretKey secret_key(context);
  PhantomPublicKey public_key = secret_key.gen_publickey(context);

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Encryptor encryptor(&context, &public_key);
  Decryptor decryptor(&context, &secret_key);
  Evaluator evaluator(&context, &phantom_encoder);

  size_t slot_count = encoder.slot_count();

  vector<double> sparse(sparse_slots, 0.0);
  vector<double> input(slot_count, 0.0);
  vector<double> before(slot_count, 0.0);
  vector<double> after(slot_count, 0.0);

  int num = 2;
  PhantomPlaintext plain;
  vector<PhantomCiphertext> cipher(num);
  // PhantomPlaintext plain;
  // PhantomCiphertext cipher;

  // Create input cipher
  for (size_t i = 0; i < slot_count; i++)
  {
    input[i] = sparse[i % sparse_slots];
  }

  for (int j = 0; j < num; j++)
  {
    encoder.encode(input, scale, plain);
    encryptor.encrypt(plain, cipher[j]);

    // Mod switch to the lowest level
    for (int i = 0; i < total_level; i++)
    {
      evaluator.mod_switch_to_next_inplace(cipher[j]);
    }
  }

  // Decrypt input cipher to obtain the original input
  decryptor.decrypt(cipher[0], plain);
  encoder.decode(plain, before);

  // ckks_evaluator.encryptor.encrypt(plain, cipher);

  // Mod switch to the lowest level
  // for (int i = 0; i < total_level; i++)
  // {
  //   ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
  // }

  // // Decrypt input cipher to obtain the original input
  // ckks_evaluator.decryptor.decrypt(cipher, plain);
  // ckks_evaluator.encoder.decode(plain, before);
  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, num));

  // // —— 准备每线程一个流（拥有型 wrapper） —— //
  // if (stream_pool.size() < static_cast<size_t>(nthreads))
  // {
  //   stream_pool.reserve(nthreads);
  //   for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
  //   {
  //     stream_pool.emplace_back(); // 默认构造：内部创建并持有一个新 CUDA 流
  //   }
  // }
  // if (nthreads == 1)
  // {
  //   stream_pool[0] = *phantom::util::global_variables::default_stream;
  // }

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  // 可选：单独的计时流，避免用默认流 0
  cudaStream_t timing_stream = nullptr;
  cudaStreamCreateWithFlags(&timing_stream, cudaStreamNonBlocking);

  vector<PhantomCiphertext> rtn(num);

  auto start = system_clock::now();

  // PhantomCiphertext rtn;
  // bootstrapper.bootstrap_3(rtn, cipher);
// CUDA_API_PER_THREAD_DEFAULT_STREAM=1
#pragma omp parallel num_threads(nthreads)
  {

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();

    Bootstrapper bootstrapper(
        loge,
        logn,
        logN - 1,
        total_level,
        scale,
        boundary_K,
        deg,
        scale_factor,
        inverse_deg,
        &ckks_evaluator);

    std::cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();

    std::cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++)
    {
      gal_steps_vector.push_back((1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

    std::cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();

    random_real(sparse, sparse_slots);
    // const int tid = omp_get_thread_num();
    // auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

    cudaStreamWaitEvent(phantom::util::global_variables::default_stream->get_stream(), ev_start, 0);
    // 确保所有线程都已经设置好 wait 之后再开枪
#pragma omp barrier
#pragma omp single
    {
      // 起跑枪：现在才开始计时，预处理不包含
      cudaEventRecord(ev_start, timing_stream ? timing_stream : 0);
    }
#pragma omp for schedule(static)
    for (int i = 0; i < num; i++)
    {
      bootstrapper.bootstrap_3(rtn[i], cipher[i]);
    }
    cudaStreamSynchronize(phantom::util::global_variables::default_stream->get_stream());
  }

  // 在并行区外创建/记录 ev_done 更清晰：每个线程结束前在各自流上 Record
  std::vector<cudaEvent_t> ev_done(nthreads);
  for (int i = 0; i < nthreads; ++i)
  {
    cudaEventCreateWithFlags(&ev_done[i], cudaEventDisableTiming);
    cudaEventRecord(ev_done[i], phantom::util::global_variables::default_stream->get_stream());
  }

  // 聚合所有 done 到计时流
  for (int i = 0; i < nthreads; ++i)
  {
    cudaStreamWaitEvent(timing_stream ? timing_stream : 0, ev_done[i], 0);
  }

  // 记录 stop 并计算时间
  cudaEventRecord(ev_stop, timing_stream ? timing_stream : 0);
  cudaEventSynchronize(ev_stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  cout << "bt compute time = " << ms << " ms\n";

  duration<double> sec = system_clock::now() - start;
  std::cout << "Bootstrapping took: " << sec.count() / num << "s" << endl;
  std::cout << "Return cipher level: " << rtn[0].coeff_modulus_size() << endl;

  // cudaError_t err = cudaDeviceSynchronize();  // <<<<<< 加这里
  // if (err != cudaSuccess) {
  //     std::cerr << "CUDA error after bootstrap: " << cudaGetErrorString(err) << std::endl;
  // }

  decryptor.decrypt(rtn[0], plain);
  encoder.decode(plain, after);

  double mean_err = 0;
  for (long i = 0; i < sparse_slots; i++)
  {
    // if (i < 10) std::cout << before[i] << " <----> " << after[i] << endl;
    mean_err += abs(before[i] - after[i]);
  }
  mean_err /= sparse_slots;
  std::cout << "Mean absolute error: " << mean_err << endl;
}
