#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

vector<PhantomCiphertext> RMSNorm(const vector<PhantomCiphertext> &x, const vector<double> &gamma,
                                    PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk)
{

    // algorithm may be different for different data range
    //RMSNorm = x / sqrt((x0^2+x1^2+...+x4096^2)/4096 + eps) * gamma
    //        = x * sqrt_inv((x0^2+x1^2+...+x4096^2)/ d^2 + eps / d) * gamma / sqrt(d)
    PhantomCKKSEncoder phantom_encoder(context);

    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    // for test
    Decryptor decryptor(&context, &sk);

    double scale = x[0].scale();
    size_t slot_count = encoder.slot_count();
    int num_ct = x.size();
    if (num_ct != 4096)
    {
        cout << "ERROR: INPUT SIZE IS NOT CORRECT. " << endl;
    }

    vector<PhantomCiphertext> output(num_ct);
    // var = (x0^2+x1^2+...+x4096^2)
    PhantomCiphertext var;
    for (int i = 0; i < num_ct; ++i)
    {   
        if (i == 0){
            PhantomCiphertext temp = x[i];
            evaluator.square_inplace(temp);
            var = temp;
        }
        else{
            PhantomCiphertext temp = x[i];
            evaluator.square_inplace(temp);
            evaluator.add_inplace(var, temp);
        }
    }

    evaluator.relinearize_inplace(var, relin_keys);
    evaluator.rescale_to_next_inplace(var);
    // var.scale() = scale;

    vector<double> ecd_inv_n2(slot_count, 0);
    for (int i = 0; i < slot_count; ++i)
    {
        ecd_inv_n2[i] = (1 / (4096.0 * 4096.0) )*5000;
    }
    PhantomPlaintext inv_d;
    encoder.encode(ecd_inv_n2, var.params_id(), var.scale(), inv_d);
    evaluator.multiply_plain_inplace(var, inv_d);
    evaluator.rescale_to_next_inplace(var);


    double eps = 1e-5;
    PhantomPlaintext plain_eps;
    encoder.encode(5000 * eps / 4096.0, var.params_id(), var.scale(), plain_eps);
    evaluator.add_plain_inplace(var, plain_eps);

    //DEBUG
    // cout <<"  decrypt of var: "<<endl;
    // PhantomPlaintext plain_result;
    // vector<double> result;
    // decryptor.decrypt(var,plain_result);
    // encoder.decode(plain_result,result);
    // for (int ind = 0 ; ind < 10 ; ++ind){
    //     cout <<result[ind]<<" ";
    // }
    // cout <<endl;

    // compute 1/sqrt(var)
    PhantomCiphertext inv_sqrt_var = invert_sqrt(var, 4, 2, context, relin_keys);

    //DEBUG
    // cout <<"  decrypt of inv_sqrt_var: "<<endl;
    // decryptor.decrypt(inv_sqrt_var,plain_result);
    // encoder.decode(plain_result,result);
    // for (int ind = 0 ; ind < 10 ; ++ind){
    //     cout <<result[ind]<<" ";
    // }
    // cout <<endl;

    // multiply x * inv_sqrt_var * gamma 
    for (int i = 0; i < num_ct; ++i)
    {
        output[i] = x[i];
        evaluator.mod_switch_to_inplace(output[i], inv_sqrt_var.params_id());
        evaluator.multiply_inplace(output[i], inv_sqrt_var);
        evaluator.relinearize_inplace(output[i], relin_keys);
        evaluator.rescale_to_next_inplace(output[i]);
        
        //DEBUG
        // cout <<"  decrypt of x * inv_sqrt_var before mul gamma: "<<endl;
        // decryptor.decrypt(output[i],plain_result);
        // encoder.decode(plain_result,result);
        // for (int ind = 0 ; ind < 10 ; ++ind){
        //     cout <<result[ind]<<" ";
        // }
        // cout <<endl;

    }
    // gamma
    vector<vector<double>> vec_gamma(4096, vector<double>(slot_count, 0));
    for (int j = 0; j < 4096; ++j){
        for (int i = 0; i < slot_count; ++i){
          vec_gamma[j][i] = sqrt(5000)* gamma[j] / sqrt(4096.0);
      }
    }

    for (int i = 0; i < num_ct; ++i)
    {
        PhantomPlaintext ecd_gamma;
        encoder.encode(vec_gamma[i], var.params_id(), var.scale(), ecd_gamma);
        evaluator.mod_switch_to_inplace(ecd_gamma, output[i].params_id());
        evaluator.multiply_plain_inplace(output[i], ecd_gamma);
        evaluator.rescale_to_next_inplace(output[i]);
    }
 
  return output;
}

vector<PhantomCiphertext> RMSNorm2(vector<PhantomCiphertext> &x, vector<double> &gamma, vector<double> &beta, const vector<int> &bias_vec,
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