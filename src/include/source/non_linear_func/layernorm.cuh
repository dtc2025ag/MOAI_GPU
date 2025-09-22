#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

// PhantomCiphertext evalLine(PhantomCiphertext x, PhantomPlaintext m, PhantomPlaintext c, PhantomContext &context)
// {
//   PhantomCKKSEncoder encoder(context);
//   Evaluator evaluator(&context, &encoder);
//   double scale = x.scale();

//   evaluator.mod_switch_to_inplace(m, x.params_id());
//   evaluator.multiply_plain_inplace(x, m);
//   evaluator.rescale_to_next_inplace(x);
//   evaluator.mod_switch_to_inplace(c, x.params_id());
//   x.scale() = scale;
//   evaluator.add_plain_inplace(x, c);
//   return x;
// }

// PhantomCiphertext initGuess(PhantomCiphertext x, PhantomContext &context)
// {
//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   PhantomPlaintext a, b;
//   encoder.encode(-1.29054537e-04, x.scale(), a);
//   encoder.encode(1.29054537e-01, x.scale(), b);
//   return evalLine(x, a, b, context);
// }

// PhantomCiphertext newtonIter(PhantomCiphertext x, PhantomCiphertext res, int iter,
//                              PhantomContext &context, PhantomRelinKey &relin_keys)
// {
//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   double scale = x.scale();

//   for (int i = 0; i < iter; ++i)
//   {
//     PhantomPlaintext three_half, neg_half;
//     encoder.encode(1.5, scale, three_half);
//     encoder.encode(-0.5, scale, neg_half);

//     // x^2
//     PhantomCiphertext res_sq;
//     evaluator.square(res, res_sq);
//     evaluator.relinearize_inplace(res_sq, relin_keys);
//     evaluator.rescale_to_next_inplace(res_sq);

//     //-0.5*x*b
//     PhantomCiphertext res_x;
//     evaluator.mod_switch_to_inplace(neg_half, x.params_id());
//     evaluator.multiply_plain(x, neg_half, res_x);
//     evaluator.rescale_to_next_inplace(res_x);
//     if (context.get_context_data(res.params_id()).chain_depth() <
//         context.get_context_data(res_x.params_id()).chain_depth())
//     {
//       evaluator.mod_switch_to_inplace(res_x, res.params_id());
//     }
//     else
//     {
//       evaluator.mod_switch_to_inplace(res, res_x.params_id());
//     }

//     evaluator.multiply_inplace(res_x, res);
//     evaluator.relinearize_inplace(res_x, relin_keys);
//     evaluator.rescale_to_next_inplace(res_x);

//     //-0.5*b*x^3
//     evaluator.mod_switch_to_inplace(res_sq, res_x.params_id());
//     evaluator.multiply_inplace(res_x, res_sq);
//     evaluator.relinearize_inplace(res_x, relin_keys);
//     evaluator.rescale_to_next_inplace(res_x);

//     // 1.5*x
//     evaluator.mod_switch_to_inplace(three_half, res.params_id());
//     evaluator.multiply_plain_inplace(res, three_half);
//     evaluator.rescale_to_next_inplace(res);

//     //-0.5*b*x^3 + 1.5*x
//     evaluator.mod_switch_to_inplace(res, res_x.params_id());
//     res_x.scale() = scale;
//     res.scale() = scale;
//     evaluator.add_inplace(res, res_x);
//   }
//   return res;
// }

// PhantomCiphertext goldSchmidtIter(PhantomCiphertext v, PhantomCiphertext y, int d,
//                                   PhantomContext &context, PhantomRelinKey &relin_keys)
// {
//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   double scale = y.scale();
//   // cout <<"scale = "<<log2(scale)<<endl;

//   PhantomPlaintext constant;
//   encoder.encode(0.5, scale, constant);

//   // GoldSchmidt's algorithm
//   evaluator.mod_switch_to_inplace(v, y.params_id());
//   PhantomCiphertext x;
//   evaluator.multiply(v, y, x);
//   evaluator.relinearize_inplace(x, relin_keys);
//   evaluator.rescale_to_next_inplace(x);

//   evaluator.mod_switch_to_inplace(constant, y.params_id());
//   PhantomCiphertext h;
//   evaluator.multiply_plain(y, constant, h);
//   evaluator.rescale_to_next_inplace(h);

//   for (int i = 0; i < d; ++i)
//   {
//     encoder.encode(0.5, scale, constant);
//     PhantomCiphertext r;
//     evaluator.multiply(x, h, r);
//     evaluator.relinearize_inplace(r, relin_keys);
//     evaluator.rescale_to_next_inplace(r);
//     r.scale() = scale;

//     PhantomCiphertext temp;
//     evaluator.negate(r, temp);
//     evaluator.mod_switch_to_inplace(constant, temp.params_id());
//     evaluator.add_plain(temp, constant, r);

//     // x = x + x*r
//     evaluator.mod_switch_to_inplace(x, r.params_id());
//     evaluator.multiply(x, r, temp);
//     evaluator.relinearize_inplace(temp, relin_keys);
//     evaluator.rescale_to_next_inplace(temp);
//     x.scale() = scale;
//     temp.scale() = scale;
//     evaluator.mod_switch_to_inplace(x, temp.params_id());
//     evaluator.add_inplace(x, temp);

//     // h = h + h*r
//     evaluator.mod_switch_to_inplace(h, r.params_id());
//     evaluator.multiply(h, r, temp);
//     evaluator.relinearize_inplace(temp, relin_keys);
//     evaluator.rescale_to_next_inplace(temp);
//     h.scale() = scale;
//     temp.scale() = scale;
//     evaluator.mod_switch_to_inplace(h, temp.params_id());
//     evaluator.add_inplace(h, temp);
//   }
//   encoder.encode(2.0, scale, constant);
//   evaluator.mod_switch_to_inplace(constant, h.params_id());
//   evaluator.multiply_plain_inplace(h, constant);
//   evaluator.rescale_to_next_inplace(h);

//   return h;
// }

// PhantomCiphertext invert_sqrt(PhantomCiphertext x, int d_newt, int d_gold,
//                               PhantomContext &context, PhantomRelinKey &relin_keys)
// {
//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   PhantomCiphertext res = initGuess(x, context);
//   PhantomCiphertext y = newtonIter(x, res, d_newt, context, relin_keys);
//   PhantomCiphertext sqrt_inv = goldSchmidtIter(x, y, d_gold, context, relin_keys);
//   return sqrt_inv;
// }

// vector<PhantomCiphertext> layernorm(const vector<PhantomCiphertext> &x, const vector<double> &gamma, const vector<double> &beta, const vector<int> &bias_vec,
//                                     PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk)
// {
//   // algorithm may be different for different data range
//   // depth need = 20 (current version)
//   PhantomCKKSEncoder phantom_encoder(context);

//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   // for test
//   Decryptor decryptor(&context, &sk);

//   double scale = x[0].scale();
//   size_t slot_count = encoder.slot_count();
//   int num_ct = x.size();
//   if (num_ct != 768)
//   {
//     cout << "ERROR: INPUT SIZE IS NOT CORRECT. " << endl;
//   }

//   // compute u=(x0+x1+...+x768)
//   PhantomCiphertext ave_x = x[0];
//   for (int i = 1; i < num_ct; ++i)
//   {
//     evaluator.add_inplace(ave_x, x[i]);
//   }
//   // evaluator.multiply_plain_inplace(ave_x,d);
//   // evaluator.rescale_to_next_inplace(ave_x);
//   // cout <<"scale of ave_x = "<<log2(ave_x.scale())<<endl;
//   //  cout <<"Modulus chain index for ave_x: "<<
//   //    seal_context.get_context_data(ave_x.parms_id())->params_id()<<endl;

//   /*
//     //for test
//     Plaintext plain_result;
//     vector<double> result;
//     cout <<"  decrypt of sum_x: "<<endl;;
//     decryptor.decrypt(ave_x,plain_result);
//     encoder.decode(plain_result,result);
//     for (int ind = 0 ; ind < slot_count ; ++ind){
//       if(bias_vec[ind] == 1){
//           cout <<result[ind]<<" ";
//       }
//     }
//     cout <<endl;

//   */
//   // compute nx_0,...,nx_n
//   vector<PhantomCiphertext> nx(num_ct);

//   const int max_threads = omp_get_max_threads();
//   const int nthreads = std::max(1, std::min(max_threads, 32));

//   if (stream_pool.size() < static_cast<size_t>(nthreads))
//   {
//     stream_pool.reserve(nthreads);
//     for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
//     {
//       stream_pool.emplace_back(); 
//     }
//   }
//   if (nthreads == 1)
//   {
//     stream_pool[0] = *phantom::util::global_variables::default_stream;
//   }

// // #pragma omp parallel for
// #pragma omp parallel num_threads(nthreads)
//   {
//     PhantomCKKSEncoder phantom_encoder_local(context);
//     moai::Encoder encoder_local(&context, &phantom_encoder_local);
//     moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

//     const int tid = omp_get_thread_num();
//     auto &stream = stream_pool[tid]; 

//     PhantomPlaintext d;
//     vector<double> ecd_n(slot_count, 0);
//     for (int i = 0; i < slot_count; ++i)
//     {
//       if (bias_vec[i] == 1)
//       {
//         ecd_n[i] = 768.0;
//       }
//     }
//     encoder.encode(ecd_n, x[0].params_id(), x[0].scale(), d, stream);
//     bridge_to_default(stream);

// #pragma omp for schedule(static)
//     for (int i = 0; i < num_ct; ++i)
//     {
//       // nx[i] = x[i];
//       PhantomCiphertext nxi = deep_copy_cipher(x[i], context, stream);
//       evaluator_local.multiply_plain_inplace(nxi, d, stream);
//       evaluator_local.rescale_to_next_inplace(nxi, stream);
//       nxi.scale() = scale;
//       nx[i] = std::move(nxi);
//     }
//     cudaStreamSynchronize(stream.get_stream());
//   }

//   //  cout <<"Modulus chain index for nx1: "<<
//   //    seal_context.get_context_data(nx[1].parms_id())->params_id()<<endl;

//   /*
//     cout <<"  decrypt of nx: "<<endl;
//     for (int i = 0; i < num_ct; ++i){
//       decryptor.decrypt(nx[i], plain_result);

//       encoder.decode(plain_result, result);
//       cout <<i<<"-th: ";
//       for (int ind = 0 ; ind < slot_count ; ++ind){
//         if(bias_vec[ind] == 1){
//             cout <<result[ind]<<" ";
//         }
//       }
//     cout <<endl;
//     }
//   */
//   // compute var=((nx0-u)^2+...+(nx768-u)^2)/n^2
//   // Ciphertext var = nx[0];
//   evaluator.mod_switch_to_inplace(ave_x, nx[0].params_id());
//   // cout <<"var scale = "<<log2(var.scale())<<", ave scale = "<<log2(ave_x.scale())<<endl;
//   // var.scale()=scale;
//   ave_x.scale() = scale;
//   // evaluator.sub_inplace(var,ave_x);
//   // evaluator.square_inplace(var);
//   // evaluator.relinearize_inplace(var,relin_keys);
//   // evaluator.rescale_to_next_inplace(var);

//   // 768 = 48*16, designed for multi-thread
//   vector<PhantomCiphertext> temp_var(48);

//   // #pragma omp parallel for
//   // #pragma omp parallel num_threads(nthreads)
//   //   {
//   //     PhantomCKKSEncoder phantom_encoder_local(context);
//   //     moai::Encoder encoder_local(&context, &phantom_encoder_local);
//   //     moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

//   //     const int tid = omp_get_thread_num();
//   //     auto &stream = stream_pool[tid]; 
//   // #pragma omp for schedule(static)
//   for (int i = 0; i < 48; ++i)
//   {
//     PhantomCiphertext temp_i;
//     for (int j = 0; j < 16; ++j)
//     {
//       // cout <<i<<" ";
//       PhantomCiphertext temp = nx[i * 16 + j];
//       PhantomCiphertext ave_x_copy = deep_copy_cipher(ave_x, context);
//       // assert(ave_x_copy.data_ptr().get_stream() == temp.data_ptr().get_stream());
//       evaluator.sub_inplace(temp, ave_x_copy);
//       evaluator.square_inplace(temp);
//       // evaluator.relinearize_inplace(temp,relin_keys);
//       // evaluator.rescale_to_next_inplace(temp);
//       if (j == 0)
//       {
//         temp_i = temp;
//       }
//       else
//       {
//         evaluator.add_inplace(temp_i, temp);
//       }
//     }
//     temp_var[i] = temp_i;
//   }
//   //   cudaStreamSynchronize(stream.get_stream());
//   // }

//   PhantomCiphertext var = temp_var[0];
//   for (int i = 1; i < 48; ++i)
//   {
//     evaluator.add_inplace(var, temp_var[i]);
//   }
//   evaluator.relinearize_inplace(var, relin_keys);
//   evaluator.rescale_to_next_inplace(var);

//   vector<double> ecd_inv_n2(slot_count, 0);
//   for (int i = 0; i < slot_count; ++i)
//   {
//     if (bias_vec[i] == 1)
//     {
//       ecd_inv_n2[i] = 1 / (768.0 * 768.0);
//     }
//   }
//   PhantomPlaintext inv_d;
//   encoder.encode(ecd_inv_n2, var.params_id(), var.scale(), inv_d);
//   evaluator.multiply_plain_inplace(var, inv_d);
//   evaluator.rescale_to_next_inplace(var);

//   /*
//     Ciphertext var2 = temp_var[24];
//     for (int i = 25; i < 48; ++i){
//       evaluator.add_inplace(var2,temp_var[i]);
//     }
//     evaluator.relinearize_inplace(var2,relin_keys);
//     evaluator.rescale_to_next_inplace(var2);

//     Plaintext inv_d2;
//     encoder.encode(ecd_inv_n2, var2.parms_id(), var2.scale(), inv_d2);
//     evaluator.multiply_plain_inplace(var2,inv_d2);
//     evaluator.rescale_to_next_inplace(var2);
//   */
//   // evaluator.add_inplace(var,var2);

//   // cout << "Modulus chain index for var: " << context.get_context_data(var.params_id()).chain_depth() << endl;
//   /*
//     cout <<"  decrypt of var: "<<endl;
//     Plaintext plain_result;
//     vector<double> result;
//     decryptor.decrypt(var,plain_result);
//     encoder.decode(plain_result,result);
//     for (int ind = 0 ; ind < slot_count ; ++ind){
//       if(bias_vec[ind] == 1){
//           cout <<result[ind]<<" ";
//       }
//     }
//     cout <<endl;
//   */
//   // compute 1/sqrt(var)
//   PhantomCiphertext inv_sqrt_var = invert_sqrt(var, 4, 2, context, relin_keys);
//   cudaDeviceSynchronize();

//   // for test
//   //  cout <<"Modulus chain index for invert sqrt: "<<
//   //    seal_context.get_context_data(inv_sqrt_var.parms_id())->params_id()<<endl;
//   /*
//   cout <<"  decrypt of 1/sqrt(var): "<<endl;;
//   decryptor.decrypt(inv_sqrt_var,plain_result);
//   encoder.decode(plain_result,result);
//   for (int ind = 0 ; ind < slot_count ; ++ind){
//     if(bias_vec[ind] == 1){
//         cout <<result[ind]<<" ";
//     }
//   }
//   cout <<endl;
// */
//   // compute Gamma/sqrt(n)*(nxi-u)*inv+beta
//   vector<PhantomCiphertext> output(num_ct);
//   evaluator.mod_switch_to_inplace(ave_x, inv_sqrt_var.params_id());

//   // #pragma omp parallel for
//   // #pragma omp parallel num_threads(nthreads)
//   //   {
//   //     PhantomCKKSEncoder phantom_encoder_local(context);
//   //     moai::Encoder encoder_local(&context, &phantom_encoder_local);
//   //     moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

//   //     const int tid = omp_get_thread_num();
//   //     auto &stream = stream_pool[tid];

//   // #pragma omp for schedule(static)
//   for (int i = 0; i < num_ct; ++i)
//   {
//     // cout<<i<<" ";
//     output[i] = nx[i];
//     evaluator.mod_switch_to_inplace(output[i], inv_sqrt_var.params_id());
//     evaluator.sub_inplace(output[i], ave_x);
//     evaluator.multiply_inplace(output[i], inv_sqrt_var);
//     evaluator.relinearize_inplace(output[i], relin_keys);
//     evaluator.rescale_to_next_inplace(output[i]);

//     vector<double> ecd_gamma_n(slot_count, 0);
//     for (int j = 0; j < slot_count; ++j)
//     {
//       if (bias_vec[j] == 1)
//       {
//         ecd_gamma_n[j] = gamma[i] / sqrt(768.0);
//       }
//     }
//     PhantomPlaintext ecd_gamma;
//     encoder.encode(ecd_gamma_n, output[i].params_id(), output[i].scale(), ecd_gamma);
//     // bridge_to_default(stream);
//     evaluator.multiply_plain_inplace(output[i], ecd_gamma);
//     evaluator.rescale_to_next_inplace(output[i]);

//     vector<double> ecd_betai(slot_count, 0);
//     for (int j = 0; j < slot_count; ++j)
//     {
//       if (bias_vec[j] == 1)
//       {
//         ecd_betai[j] = beta[i];
//       }
//     }
//     PhantomPlaintext ecd_beta;
//     encoder.encode(ecd_betai, output[i].params_id(), output[i].scale(), ecd_beta);
//     // bridge_to_default(stream);
//     evaluator.add_plain_inplace(output[i], ecd_beta);
//   }
//   //   cudaStreamSynchronize(stream.get_stream());
//   // }

//   return output;
// }

// vector<PhantomCiphertext> layernorm2(vector<PhantomCiphertext> &x, vector<double> &gamma, vector<double> &beta, const vector<int> &bias_vec,
//                                      PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk)
// {
//   // algorithm may be different for different data range
//   // depth need = 20 (current version)
//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);
//   // for test
//   Decryptor decryptor(&context, &sk);

//   double scale = x[0].scale();
//   size_t slot_count = encoder.slot_count();
//   int num_ct = x.size();
//   if (num_ct != 768)
//   {
//     cout << "ERROR: INPUT SIZE IS NOT CORRECT. " << endl;
//   }

//   // compute u=(x0+x1+...+x768)
//   PhantomCiphertext ave_x = x[0];
//   for (int i = 1; i < num_ct; ++i)
//   {
//     evaluator.add_inplace(ave_x, x[i]);
//   }
//   // evaluator.multiply_plain_inplace(ave_x,d);
//   // evaluator.rescale_to_next_inplace(ave_x);
//   // cout <<"scale of ave_x = "<<log2(ave_x.scale())<<endl;
//   //  cout <<"Modulus chain index for ave_x: "<<
//   //    seal_context.get_context_data(ave_x.parms_id())->params_id()<<endl;

//   /*
//     //for test
//     Plaintext plain_result;
//     vector<double> result;
//     cout <<"  decrypt of sum_x: "<<endl;;
//     decryptor.decrypt(ave_x,plain_result);
//     encoder.decode(plain_result,result);
//     for (int ind = 0 ; ind < slot_count ; ++ind){
//       if(bias_vec[ind] == 1){
//           cout <<result[ind]<<" ";
//       }
//     }
//     cout <<endl;
//   */

//   // compute nx_0,...,nx_n
//   PhantomPlaintext d;
//   vector<double> ecd_n(slot_count, 0);
//   for (int i = 0; i < slot_count; ++i)
//   {
//     if (bias_vec[i] == 1)
//     {
//       ecd_n[i] = 768.0;
//     }
//   }
//   encoder.encode(ecd_n, x[0].params_id(), x[0].scale(), d);
//   vector<PhantomCiphertext> nx(num_ct);

//   // #pragma omp parallel for

//   for (int i = 0; i < num_ct; ++i)
//   {
//     nx[i] = x[i];
//     evaluator.multiply_plain_inplace(nx[i], d);
//     evaluator.rescale_to_next_inplace(nx[i]);
//     nx[i].scale() = scale;
//   }

//   //  cout <<"Modulus chain index for nx1: "<<
//   //    seal_context.get_context_data(nx[1].parms_id())->params_id()<<endl;

//   /*
//     cout <<"  decrypt of nx: "<<endl;
//     for (int i = 0; i < num_ct; ++i){
//       decryptor.decrypt(nx[i], plain_result);

//       encoder.decode(plain_result, result);
//       cout <<i<<"-th: ";
//       for (int ind = 0 ; ind < slot_count ; ++ind){
//         if(bias_vec[ind] == 1){
//             cout <<result[ind]<<" ";
//         }
//       }
//     cout <<endl;
//     }
//   */
//   // compute var=((nx0-u)^2+...+(nx768-u)^2)/n^2
//   // Ciphertext var = nx[0];
//   evaluator.mod_switch_to_inplace(ave_x, nx[0].params_id());
//   // cout <<"var scale = "<<log2(var.scale())<<", ave scale = "<<log2(ave_x.scale())<<endl;
//   // var.scale()=scale;
//   ave_x.scale() = scale;
//   // evaluator.sub_inplace(var,ave_x);
//   // evaluator.square_inplace(var);
//   // evaluator.relinearize_inplace(var,relin_keys);
//   // evaluator.rescale_to_next_inplace(var);

//   // 768 = 48*16, designed for multi-thread
//   vector<PhantomCiphertext> temp_var(48);

//   // #pragma omp parallel for

//   for (int i = 0; i < 48; ++i)
//   {
//     PhantomCiphertext temp_i;
//     for (int j = 0; j < 16; ++j)
//     {
//       // cout <<i<<" ";
//       PhantomCiphertext temp = nx[i * 16 + j];
//       evaluator.sub_inplace(temp, ave_x);
//       evaluator.square_inplace(temp);
//       // evaluator.relinearize_inplace(temp,relin_keys);
//       // evaluator.rescale_to_next_inplace(temp);
//       if (j == 0)
//       {
//         temp_i = temp;
//       }
//       else
//       {
//         evaluator.add_inplace(temp_i, temp);
//       }
//     }
//     temp_var[i] = temp_i;
//   }

//   PhantomCiphertext var = temp_var[0];
//   for (int i = 1; i < 48; ++i)
//   {
//     evaluator.add_inplace(var, temp_var[i]);
//   }
//   evaluator.relinearize_inplace(var, relin_keys);
//   evaluator.rescale_to_next_inplace(var);

//   vector<double> ecd_inv_n2(slot_count, 0);
//   for (int i = 0; i < slot_count; ++i)
//   {
//     if (bias_vec[i] == 1)
//     {
//       ecd_inv_n2[i] = 1 / (768.0 * 768.0 * 768.0);
//     }
//   }
//   PhantomPlaintext inv_d;
//   encoder.encode(ecd_inv_n2, var.params_id(), var.scale(), inv_d);
//   evaluator.multiply_plain_inplace(var, inv_d);
//   evaluator.rescale_to_next_inplace(var);
//   /*
//     cout <<"Modulus chain index for var: "<<
//       seal_context.get_context_data(var.parms_id())->params_id()<<endl;

//     cout <<"  decrypt of var: "<<endl;
//     Plaintext plain_result;
//     vector<double> result;
//     decryptor.decrypt(var,plain_result);
//     encoder.decode(plain_result,result);
//     for (int ind = 0 ; ind < slot_count ; ++ind){
//       if(bias_vec[ind] == 1){
//           cout <<result[ind]<<" ";
//       }
//     }
//     cout <<endl;
//   */
//   // compute 1/sqrt(var)
//   PhantomCiphertext inv_sqrt_var = invert_sqrt(var, 4, 2, context, relin_keys);
//   /*
//     //for test
//     cout <<"Modulus chain index for invert sqrt: "<<
//       seal_context.get_context_data(inv_sqrt_var.parms_id())->params_id()<<endl;

//     cout <<"  decrypt of 1/sqrt(var): "<<endl;;
//     decryptor.decrypt(inv_sqrt_var,plain_result);
//     encoder.decode(plain_result,result);
//     for (int ind = 0 ; ind < slot_count ; ++ind){
//       if(bias_vec[ind] == 1){
//           cout <<result[ind]<<" ";
//       }
//     }
//     cout <<endl;
//   */
//   // compute Gamma/sqrt(n)*(nxi-u)*inv+beta
//   vector<PhantomCiphertext> output(num_ct);
//   evaluator.mod_switch_to_inplace(ave_x, inv_sqrt_var.params_id());

//   // #pragma omp parallel for

//   for (int i = 0; i < num_ct; ++i)
//   {
//     // cout<<i<<" ";
//     output[i] = nx[i];
//     evaluator.mod_switch_to_inplace(output[i], inv_sqrt_var.params_id());
//     evaluator.sub_inplace(output[i], ave_x);
//     evaluator.multiply_inplace(output[i], inv_sqrt_var);
//     evaluator.relinearize_inplace(output[i], relin_keys);
//     evaluator.rescale_to_next_inplace(output[i]);

//     vector<double> ecd_gamma_n(slot_count, 0);
//     for (int j = 0; j < slot_count; ++j)
//     {
//       if (bias_vec[j] == 1)
//       {
//         ecd_gamma_n[j] = gamma[i] / 768.0;
//       }
//     }
//     PhantomPlaintext ecd_gamma;
//     encoder.encode(ecd_gamma_n, output[i].params_id(), output[i].scale(), ecd_gamma);
//     evaluator.multiply_plain_inplace(output[i], ecd_gamma);
//     evaluator.rescale_to_next_inplace(output[i]);

//     vector<double> ecd_betai(slot_count, 0);
//     for (int j = 0; j < slot_count; ++j)
//     {
//       if (bias_vec[j] == 1)
//       {
//         ecd_betai[j] = beta[i];
//       }
//     }
//     PhantomPlaintext ecd_beta;
//     encoder.encode(ecd_betai, output[i].params_id(), output[i].scale(), ecd_beta);
//     evaluator.add_plain_inplace(output[i], ecd_beta);
//   }

//   return output;
// }

PhantomCiphertext evalLine(PhantomCiphertext x, PhantomPlaintext m, PhantomPlaintext c, PhantomContext &context)
{
  PhantomCKKSEncoder encoder(context);
  Evaluator evaluator(&context, &encoder);
  double scale = x.scale();

  evaluator.mod_switch_to_inplace(m, x.params_id());
  evaluator.multiply_plain_inplace(x, m);
  evaluator.rescale_to_next_inplace(x);
  evaluator.mod_switch_to_inplace(c, x.params_id());
  x.scale() = scale;
  evaluator.add_plain_inplace(x, c);
  return x;
}

PhantomCiphertext initGuess(PhantomCiphertext x, PhantomContext &context)
{
  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  PhantomPlaintext a, b;
  encoder.encode(-1.29054537e-04, x.scale(), a);
  encoder.encode(1.29054537e-01, x.scale(), b);
  return evalLine(x, a, b, context);
}

PhantomCiphertext newtonIter(PhantomCiphertext x, PhantomCiphertext res, int iter,
                             PhantomContext &context, PhantomRelinKey &relin_keys)
{
  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  double scale = x.scale();

  for (int i = 0; i < iter; ++i)
  {
    PhantomPlaintext three_half, neg_half;
    encoder.encode(1.5, scale, three_half);
    encoder.encode(-0.5, scale, neg_half);

    // x^2
    PhantomCiphertext res_sq;
    evaluator.square(res, res_sq);
    evaluator.relinearize_inplace(res_sq, relin_keys);
    evaluator.rescale_to_next_inplace(res_sq);

    //-0.5*x*b
    PhantomCiphertext res_x;
    evaluator.mod_switch_to_inplace(neg_half, x.params_id());
    evaluator.multiply_plain(x, neg_half, res_x);
    evaluator.rescale_to_next_inplace(res_x);
    if (context.get_context_data(res.params_id()).chain_depth() <
        context.get_context_data(res_x.params_id()).chain_depth())
    {
      evaluator.mod_switch_to_inplace(res_x, res.params_id());
    }
    else
    {
      evaluator.mod_switch_to_inplace(res, res_x.params_id());
    }

    evaluator.multiply_inplace(res_x, res);
    evaluator.relinearize_inplace(res_x, relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    //-0.5*b*x^3
    evaluator.mod_switch_to_inplace(res_sq, res_x.params_id());
    evaluator.multiply_inplace(res_x, res_sq);
    evaluator.relinearize_inplace(res_x, relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    // 1.5*x
    evaluator.mod_switch_to_inplace(three_half, res.params_id());
    evaluator.multiply_plain_inplace(res, three_half);
    evaluator.rescale_to_next_inplace(res);

    //-0.5*b*x^3 + 1.5*x
    evaluator.mod_switch_to_inplace(res, res_x.params_id());
    res_x.scale() = scale;
    res.scale() = scale;
    evaluator.add_inplace(res, res_x);
  }
  return res;
}

PhantomCiphertext goldSchmidtIter(PhantomCiphertext v, PhantomCiphertext y, int d,
                                  PhantomContext &context, PhantomRelinKey &relin_keys)
{
  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  double scale = y.scale();
  // cout <<"scale = "<<log2(scale)<<endl;

  PhantomPlaintext constant;
  encoder.encode(0.5, scale, constant);

  // GoldSchmidt's algorithm
  evaluator.mod_switch_to_inplace(v, y.params_id());
  PhantomCiphertext x;
  evaluator.multiply(v, y, x);
  evaluator.relinearize_inplace(x, relin_keys);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(constant, y.params_id());
  PhantomCiphertext h;
  evaluator.multiply_plain(y, constant, h);
  evaluator.rescale_to_next_inplace(h);

  for (int i = 0; i < d; ++i)
  {
    encoder.encode(0.5, scale, constant);
    PhantomCiphertext r;
    evaluator.multiply(x, h, r);
    evaluator.relinearize_inplace(r, relin_keys);
    evaluator.rescale_to_next_inplace(r);
    r.scale() = scale;

    PhantomCiphertext temp;
    evaluator.negate(r, temp);
    evaluator.mod_switch_to_inplace(constant, temp.params_id());
    evaluator.add_plain(temp, constant, r);

    // x = x + x*r
    evaluator.mod_switch_to_inplace(x, r.params_id());
    evaluator.multiply(x, r, temp);
    evaluator.relinearize_inplace(temp, relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    x.scale() = scale;
    temp.scale() = scale;
    evaluator.mod_switch_to_inplace(x, temp.params_id());
    evaluator.add_inplace(x, temp);

    // h = h + h*r
    evaluator.mod_switch_to_inplace(h, r.params_id());
    evaluator.multiply(h, r, temp);
    evaluator.relinearize_inplace(temp, relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    h.scale() = scale;
    temp.scale() = scale;
    evaluator.mod_switch_to_inplace(h, temp.params_id());
    evaluator.add_inplace(h, temp);
  }
  encoder.encode(2.0, scale, constant);
  evaluator.mod_switch_to_inplace(constant, h.params_id());
  evaluator.multiply_plain_inplace(h, constant);
  evaluator.rescale_to_next_inplace(h);

  return h;
}

PhantomCiphertext invert_sqrt(PhantomCiphertext x, int d_newt, int d_gold,
                              PhantomContext &context, PhantomRelinKey &relin_keys)
{
  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);

  PhantomCiphertext res = initGuess(x, context);
  PhantomCiphertext y = newtonIter(x, res, d_newt, context, relin_keys);
  PhantomCiphertext sqrt_inv = goldSchmidtIter(x, y, d_gold, context, relin_keys);
  return sqrt_inv;
}

vector<PhantomCiphertext> layernorm(const vector<PhantomCiphertext> &x, const vector<double> &gamma, const vector<double> &beta, const vector<int> &bias_vec,
                                    PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk)
{
  // algorithm may be different for different data range
  // depth need = 20 (current version)
  PhantomCKKSEncoder phantom_encoder(context);

  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  // for test
  Decryptor decryptor(&context, &sk);

  double scale = x[0].scale();
  size_t slot_count = encoder.slot_count();
  int num_ct = x.size();
  if (num_ct != 768)
  {
    cout << "ERROR: INPUT SIZE IS NOT CORRECT. " << endl;
  }

  // compute u=(x0+x1+...+x768)
  PhantomCiphertext ave_x = x[0];
  for (int i = 1; i < num_ct; ++i)
  {
    evaluator.add_inplace(ave_x, x[i]);
  }
  // evaluator.multiply_plain_inplace(ave_x,d);
  // evaluator.rescale_to_next_inplace(ave_x);
  // cout <<"scale of ave_x = "<<log2(ave_x.scale())<<endl;
  //  cout <<"Modulus chain index for ave_x: "<<
  //    seal_context.get_context_data(ave_x.parms_id())->params_id()<<endl;

  /*
    //for test
    Plaintext plain_result;
    vector<double> result;
    cout <<"  decrypt of sum_x: "<<endl;;
    decryptor.decrypt(ave_x,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<" ";
      }
    }
    cout <<endl;

  */
  // compute nx_0,...,nx_n
  PhantomPlaintext d;
  vector<double> ecd_n(slot_count, 0);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      ecd_n[i] = 768.0;
    }
  }
  encoder.encode(ecd_n, x[0].params_id(), x[0].scale(), d);
  vector<PhantomCiphertext> nx(num_ct);

  // #pragma omp parallel for
  for (int i = 0; i < num_ct; ++i)
  {
    nx[i] = x[i];
    evaluator.multiply_plain_inplace(nx[i], d);
    evaluator.rescale_to_next_inplace(nx[i]);
    nx[i].scale() = scale;
  }

  //  cout <<"Modulus chain index for nx1: "<<
  //    seal_context.get_context_data(nx[1].parms_id())->params_id()<<endl;

  /*
    cout <<"  decrypt of nx: "<<endl;
    for (int i = 0; i < num_ct; ++i){
      decryptor.decrypt(nx[i], plain_result);

      encoder.decode(plain_result, result);
      cout <<i<<"-th: ";
      for (int ind = 0 ; ind < slot_count ; ++ind){
        if(bias_vec[ind] == 1){
            cout <<result[ind]<<" ";
        }
      }
    cout <<endl;
    }
  */
  // compute var=((nx0-u)^2+...+(nx768-u)^2)/n^2
  // Ciphertext var = nx[0];
  evaluator.mod_switch_to_inplace(ave_x, nx[0].params_id());
  // cout <<"var scale = "<<log2(var.scale())<<", ave scale = "<<log2(ave_x.scale())<<endl;
  // var.scale()=scale;
  ave_x.scale() = scale;
  // evaluator.sub_inplace(var,ave_x);
  // evaluator.square_inplace(var);
  // evaluator.relinearize_inplace(var,relin_keys);
  // evaluator.rescale_to_next_inplace(var);

  // 768 = 48*16, designed for multi-thread
  vector<PhantomCiphertext> temp_var(48);

  // #pragma omp parallel for
  for (int i = 0; i < 48; ++i)
  {
    PhantomCiphertext temp_i;
    for (int j = 0; j < 16; ++j)
    {
      // cout <<i<<" ";
      PhantomCiphertext temp = nx[i * 16 + j];
      evaluator.sub_inplace(temp, ave_x);
      evaluator.square_inplace(temp);
      // evaluator.relinearize_inplace(temp,relin_keys);
      // evaluator.rescale_to_next_inplace(temp);
      if (j == 0)
      {
        temp_i = temp;
      }
      else
      {
        evaluator.add_inplace(temp_i, temp);
      }
    }
    temp_var[i] = temp_i;
  }

  PhantomCiphertext var = temp_var[0];
  for (int i = 1; i < 48; ++i)
  {
    evaluator.add_inplace(var, temp_var[i]);
  }
  evaluator.relinearize_inplace(var, relin_keys);
  evaluator.rescale_to_next_inplace(var);

  vector<double> ecd_inv_n2(slot_count, 0);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      ecd_inv_n2[i] = 1 / (768.0 * 768.0);
    }
  }
  PhantomPlaintext inv_d;
  encoder.encode(ecd_inv_n2, var.params_id(), var.scale(), inv_d);
  evaluator.multiply_plain_inplace(var, inv_d);
  evaluator.rescale_to_next_inplace(var);

  /*
    Ciphertext var2 = temp_var[24];
    for (int i = 25; i < 48; ++i){
      evaluator.add_inplace(var2,temp_var[i]);
    }
    evaluator.relinearize_inplace(var2,relin_keys);
    evaluator.rescale_to_next_inplace(var2);

    Plaintext inv_d2;
    encoder.encode(ecd_inv_n2, var2.parms_id(), var2.scale(), inv_d2);
    evaluator.multiply_plain_inplace(var2,inv_d2);
    evaluator.rescale_to_next_inplace(var2);
  */
  // evaluator.add_inplace(var,var2);

  // cout << "Modulus chain index for var: " << context.get_context_data(var.params_id()).chain_depth() << endl;
  /*
    cout <<"  decrypt of var: "<<endl;
    Plaintext plain_result;
    vector<double> result;
    decryptor.decrypt(var,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<" ";
      }
    }
    cout <<endl;
  */
  // compute 1/sqrt(var)
  PhantomCiphertext inv_sqrt_var = invert_sqrt(var, 4, 2, context, relin_keys);

  // for test
  //  cout <<"Modulus chain index for invert sqrt: "<<
  //    seal_context.get_context_data(inv_sqrt_var.parms_id())->params_id()<<endl;
  /*
  cout <<"  decrypt of 1/sqrt(var): "<<endl;;
  decryptor.decrypt(inv_sqrt_var,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<" ";
    }
  }
  cout <<endl;
*/
  // compute Gamma/sqrt(n)*(nxi-u)*inv+beta
  vector<PhantomCiphertext> output(num_ct);
  evaluator.mod_switch_to_inplace(ave_x, inv_sqrt_var.params_id());

  // #pragma omp parallel for
  for (int i = 0; i < num_ct; ++i)
  {
    // cout<<i<<" ";
    output[i] = nx[i];
    evaluator.mod_switch_to_inplace(output[i], inv_sqrt_var.params_id());
    evaluator.sub_inplace(output[i], ave_x);
    evaluator.multiply_inplace(output[i], inv_sqrt_var);
    evaluator.relinearize_inplace(output[i], relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_gamma_n(slot_count, 0);
    for (int j = 0; j < slot_count; ++j)
    {
      if (bias_vec[j] == 1)
      {
        ecd_gamma_n[j] = gamma[i] / sqrt(768.0);
      }
    }
    PhantomPlaintext ecd_gamma;
    encoder.encode(ecd_gamma_n, output[i].params_id(), output[i].scale(), ecd_gamma);
    evaluator.multiply_plain_inplace(output[i], ecd_gamma);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_betai(slot_count, 0);
    for (int j = 0; j < slot_count; ++j)
    {
      if (bias_vec[j] == 1)
      {
        ecd_betai[j] = beta[i];
      }
    }
    PhantomPlaintext ecd_beta;
    encoder.encode(ecd_betai, output[i].params_id(), output[i].scale(), ecd_beta);
    evaluator.add_plain_inplace(output[i], ecd_beta);
  }

  return output;
}

vector<PhantomCiphertext> layernorm2(vector<PhantomCiphertext> &x, vector<double> &gamma, vector<double> &beta, const vector<int> &bias_vec,
                                     PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk)
{
  // algorithm may be different for different data range
  // depth need = 20 (current version)
  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  // for test
  Decryptor decryptor(&context, &sk);

  double scale = x[0].scale();
  size_t slot_count = encoder.slot_count();
  int num_ct = x.size();
  if (num_ct != 768)
  {
    cout << "ERROR: INPUT SIZE IS NOT CORRECT. " << endl;
  }

  // compute u=(x0+x1+...+x768)
  PhantomCiphertext ave_x = x[0];
  for (int i = 1; i < num_ct; ++i)
  {
    evaluator.add_inplace(ave_x, x[i]);
  }
  // evaluator.multiply_plain_inplace(ave_x,d);
  // evaluator.rescale_to_next_inplace(ave_x);
  // cout <<"scale of ave_x = "<<log2(ave_x.scale())<<endl;
  //  cout <<"Modulus chain index for ave_x: "<<
  //    seal_context.get_context_data(ave_x.parms_id())->params_id()<<endl;

  /*
    //for test
    Plaintext plain_result;
    vector<double> result;
    cout <<"  decrypt of sum_x: "<<endl;;
    decryptor.decrypt(ave_x,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<" ";
      }
    }
    cout <<endl;
  */

  // compute nx_0,...,nx_n
  PhantomPlaintext d;
  vector<double> ecd_n(slot_count, 0);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      ecd_n[i] = 768.0;
    }
  }
  encoder.encode(ecd_n, x[0].params_id(), x[0].scale(), d);
  vector<PhantomCiphertext> nx(num_ct);

  // #pragma omp parallel for

  for (int i = 0; i < num_ct; ++i)
  {
    nx[i] = x[i];
    evaluator.multiply_plain_inplace(nx[i], d);
    evaluator.rescale_to_next_inplace(nx[i]);
    nx[i].scale() = scale;
  }

  //  cout <<"Modulus chain index for nx1: "<<
  //    seal_context.get_context_data(nx[1].parms_id())->params_id()<<endl;

  /*
    cout <<"  decrypt of nx: "<<endl;
    for (int i = 0; i < num_ct; ++i){
      decryptor.decrypt(nx[i], plain_result);

      encoder.decode(plain_result, result);
      cout <<i<<"-th: ";
      for (int ind = 0 ; ind < slot_count ; ++ind){
        if(bias_vec[ind] == 1){
            cout <<result[ind]<<" ";
        }
      }
    cout <<endl;
    }
  */
  // compute var=((nx0-u)^2+...+(nx768-u)^2)/n^2
  // Ciphertext var = nx[0];
  evaluator.mod_switch_to_inplace(ave_x, nx[0].params_id());
  // cout <<"var scale = "<<log2(var.scale())<<", ave scale = "<<log2(ave_x.scale())<<endl;
  // var.scale()=scale;
  ave_x.scale() = scale;
  // evaluator.sub_inplace(var,ave_x);
  // evaluator.square_inplace(var);
  // evaluator.relinearize_inplace(var,relin_keys);
  // evaluator.rescale_to_next_inplace(var);

  // 768 = 48*16, designed for multi-thread
  vector<PhantomCiphertext> temp_var(48);

  // #pragma omp parallel for

  for (int i = 0; i < 48; ++i)
  {
    PhantomCiphertext temp_i;
    for (int j = 0; j < 16; ++j)
    {
      // cout <<i<<" ";
      PhantomCiphertext temp = nx[i * 16 + j];
      evaluator.sub_inplace(temp, ave_x);
      evaluator.square_inplace(temp);
      // evaluator.relinearize_inplace(temp,relin_keys);
      // evaluator.rescale_to_next_inplace(temp);
      if (j == 0)
      {
        temp_i = temp;
      }
      else
      {
        evaluator.add_inplace(temp_i, temp);
      }
    }
    temp_var[i] = temp_i;
  }

  PhantomCiphertext var = temp_var[0];
  for (int i = 1; i < 48; ++i)
  {
    evaluator.add_inplace(var, temp_var[i]);
  }
  evaluator.relinearize_inplace(var, relin_keys);
  evaluator.rescale_to_next_inplace(var);

  vector<double> ecd_inv_n2(slot_count, 0);
  for (int i = 0; i < slot_count; ++i)
  {
    if (bias_vec[i] == 1)
    {
      ecd_inv_n2[i] = 1 / (768.0 * 768.0 * 768.0);
    }
  }
  PhantomPlaintext inv_d;
  encoder.encode(ecd_inv_n2, var.params_id(), var.scale(), inv_d);
  evaluator.multiply_plain_inplace(var, inv_d);
  evaluator.rescale_to_next_inplace(var);
  /*
    cout <<"Modulus chain index for var: "<<
      seal_context.get_context_data(var.parms_id())->params_id()<<endl;

    cout <<"  decrypt of var: "<<endl;
    Plaintext plain_result;
    vector<double> result;
    decryptor.decrypt(var,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<" ";
      }
    }
    cout <<endl;
  */
  // compute 1/sqrt(var)
  PhantomCiphertext inv_sqrt_var = invert_sqrt(var, 4, 2, context, relin_keys);
  /*
    //for test
    cout <<"Modulus chain index for invert sqrt: "<<
      seal_context.get_context_data(inv_sqrt_var.parms_id())->params_id()<<endl;

    cout <<"  decrypt of 1/sqrt(var): "<<endl;;
    decryptor.decrypt(inv_sqrt_var,plain_result);
    encoder.decode(plain_result,result);
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<" ";
      }
    }
    cout <<endl;
  */
  // compute Gamma/sqrt(n)*(nxi-u)*inv+beta
  vector<PhantomCiphertext> output(num_ct);
  evaluator.mod_switch_to_inplace(ave_x, inv_sqrt_var.params_id());

  // #pragma omp parallel for

  for (int i = 0; i < num_ct; ++i)
  {
    // cout<<i<<" ";
    output[i] = nx[i];
    evaluator.mod_switch_to_inplace(output[i], inv_sqrt_var.params_id());
    evaluator.sub_inplace(output[i], ave_x);
    evaluator.multiply_inplace(output[i], inv_sqrt_var);
    evaluator.relinearize_inplace(output[i], relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_gamma_n(slot_count, 0);
    for (int j = 0; j < slot_count; ++j)
    {
      if (bias_vec[j] == 1)
      {
        ecd_gamma_n[j] = gamma[i] / 768.0;
      }
    }
    PhantomPlaintext ecd_gamma;
    encoder.encode(ecd_gamma_n, output[i].params_id(), output[i].scale(), ecd_gamma);
    evaluator.multiply_plain_inplace(output[i], ecd_gamma);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_betai(slot_count, 0);
    for (int j = 0; j < slot_count; ++j)
    {
      if (bias_vec[j] == 1)
      {
        ecd_betai[j] = beta[i];
      }
    }
    PhantomPlaintext ecd_beta;
    encoder.encode(ecd_betai, output[i].params_id(), output[i].scale(), ecd_beta);
    evaluator.add_plain_inplace(output[i], ecd_beta);
  }

  return output;
}