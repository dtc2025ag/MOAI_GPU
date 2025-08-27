#include "include.cuh"

using namespace std;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;



//Interlaced batching, column-encoded. 
//In BERT, num_X = 128, num_row = 128, num_col = 768
//Output: 768 Ciphertexts
// vector<PhantomCiphertext> batch_input(const vector<vector<vector<double>>> & X, 
//   int num_X, int num_row, int num_col, double scale,
//   PhantomContext& context, PhantomPublicKey& pk){

//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Encryptor encryptor(&context, &pk);
//   size_t slot_count = encoder.slot_count();
//   // cout <<"slot count = "<<slot_count<<endl;

//   vector<PhantomCiphertext> output(num_col);

//   // #pragma omp parallel for 
//   for (int i = 0; i < num_col; ++i){
//     vector<double> vec(slot_count,0);
//     for (int j = 0 ; j < num_X ; ++j){
//       for (int k = 0 ; k < num_row ; ++k){
//         vec[num_X*k+j] = X[j][k][i];
//       }
//     }
    
//     PhantomPlaintext ecd_vec;
//     encoder.encode(vec, scale, ecd_vec);
//     encryptor.encrypt(ecd_vec, output[i]);

//   }

//   return output;

// }

using std::vector;
using phantom::util::cuda_stream_wrapper;

#include "Ct_pt_matrix_mul.cuh"
// vector<cuda_stream_wrapper> stream_pool; // 线程私有流池

// // 把“自定义流上的完成”桥接到默认流，避免默认流读到半成品
// static inline void bridge_to_default(const cuda_stream_wrapper &sw) {
//   auto dst = phantom::util::global_variables::default_stream->get_stream();
//   if (sw.get_stream() == dst) return;                 // 同一条流就不需要桥接

//   cudaEvent_t ev;
//   cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
//   cudaEventRecord(ev, sw.get_stream());               // 在生产流上记录事件
//   cudaStreamWaitEvent(dst, ev, 0);                    // 让“库的默认流”等待事件
//   cudaEventDestroy(ev);                               // 等待已入队，销毁事件对象即可
// }

vector<PhantomCiphertext> batch_input(const vector<vector<vector<double>>> & X, 
  int num_X, int num_row, int num_col, double scale,
  PhantomContext& context, PhantomPublicKey& pk){

  const int max_threads = omp_get_max_threads();
  const int nthreads = std::max(1, std::min(max_threads, num_col));
  cout <<"Using "<<nthreads<<" threads for batch encoding and encryption."<<endl;

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


  // PhantomCKKSEncoder phantom_encoder(context);
  // Encoder encoder(&context, &phantom_encoder);
  // Encryptor encryptor(&context, &pk);
  // size_t slot_count = encoder.slot_count();
  // cout <<"slot count = "<<slot_count<<endl;

  vector<PhantomCiphertext> output(num_col);

   #pragma omp parallel num_threads(nthreads)
   {
      PhantomCKKSEncoder phantom_encoder_local(context);
      moai::Encoder   encoder_local(&context, &phantom_encoder_local);
      moai::Encryptor encryptor_local(&context, &pk);
      size_t slot_count = encoder_local.slot_count();

      const int tid = omp_get_thread_num();
      auto &stream = stream_pool[tid];              // ★ 引用，不要拷贝 wrapper

      #pragma omp for schedule(static)
      for (int i = 0; i < num_col; ++i){
      vector<double> vec(slot_count,0);
      for (int j = 0 ; j < num_X ; ++j){
        for (int k = 0 ; k < num_row ; ++k){
          vec[num_X*k+j] = X[j][k][i];
        }
      }
      
      PhantomPlaintext ecd_vec;
      encoder_local.encode(vec, scale, ecd_vec, stream);
      encryptor_local.encrypt(ecd_vec, output[i], stream);
      }
   }
  cudaDeviceSynchronize();

  return output;

}



vector<int> bias_vec(const vector<int> & lengths, int num_X, int num_row) {
  vector<int> output(num_X * num_row,0);
  for (int i=0; i<num_X; ++i) {
    // input X_i has size lengths[i] * num_col
    for (int j=0; j < lengths[i] ; ++j){
      output[j * num_X + i] = 1;
    }
  }
  return output;
}