// #include <chrono>
// using namespace chrono;
// #include "Bootstrapper.h"
// #include "ckks_evaluator.h"

// using namespace std;
// using namespace seal;
#include "include.cuh"
// #include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;


vector<PhantomCiphertext> single_att_block(vector<PhantomCiphertext> & enc_X, 
  vector<vector<double>> & WQ, vector<vector<double>> & WK,
  vector<vector<double>> & WV, vector<double> &bQ, 
  vector<double> &bK, vector<double> &bV, vector<int> &bias_vec, int input_num,
  PhantomContext& context, PhantomRelinKey &relin_keys, PhantomGaloisKey & RotK, Bootstrapper &bootstrapper_att,
  int num_batch, PhantomSecretKey & sk, int iter, int layer_id){

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  size_t slot_count = encoder.slot_count();
  //for test
  Decryptor decryptor(&context, &sk);
  double scale = enc_X[0].scale();

  int col_W = WQ[0].size();
  int num_col = enc_X.size();
  //cout <<"number of column of x = "<<num_col<<", number of column of WQ = "<<col_W<<endl;
  struct timeval tstart1, tend1;

  gettimeofday(&tstart1,NULL);
  vector<PhantomCiphertext> Q = ct_pt_matrix_mul_wo_pre(enc_X, WQ, num_col, col_W, num_col, context);
  //cout <<"Q = XW_Q. "<<endl;
//   cout <<"scale of Q = "<<Q[0].scale()<<", scale of bQ = "<<bQ[0].scale()<<endl;
  for (int i = 0; i < col_W; ++i){
    //cout <<i<<" ";
    PhantomPlaintext ecd_b_q;
    vector<double> bq_vec(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
        if(bias_vec[j] == 1){
            bq_vec[j] = bQ[i];
        }
    }
    encoder.encode(bq_vec, Q[i].params_id(), Q[i].scale(), ecd_b_q);
    //Plaintext temp = bQ[i];
    evaluator.mod_switch_to_inplace(ecd_b_q, Q[i].params_id());
    Q[i].scale() = scale;
    ecd_b_q.scale() = scale;
    //cout <<"scale of Q = "<<Q[i].scale()<<", scale of bQ = "<<temp.scale()<<endl;
    evaluator.add_plain_inplace(Q[i],ecd_b_q);
  }


  //cout <<"Q += b"<<endl;
  vector<PhantomCiphertext> K = ct_pt_matrix_mul_wo_pre(enc_X, WK, num_col, col_W, num_col, context);

  for (int i = 0; i < col_W; ++i){
    PhantomPlaintext ecd_b_k;
    vector<double> bk_vec(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
        if(bias_vec[j] == 1){
            bk_vec[j] = bK[i];
        }
    }
    encoder.encode(bk_vec, K[i].params_id(), K[i].scale(), ecd_b_k);
    //Plaintext temp = bK[i];
    evaluator.mod_switch_to_inplace(ecd_b_k, K[i].params_id());
    K[i].scale() = scale;
    ecd_b_k.scale() = scale;
    evaluator.add_plain_inplace(K[i],ecd_b_k);

  }

  vector<PhantomCiphertext> enc_X_v(num_col);

  #pragma omp parallel for

  for (int i = 0; i < num_col; ++i){
      enc_X_v[i] = enc_X[i];
      while (context.get_context_data(enc_X_v[i].params_id()).chain_depth()>3){
        evaluator.mod_switch_to_next_inplace(enc_X_v[i]);
      }
  }
  vector<PhantomCiphertext> V = ct_pt_matrix_mul_wo_pre(enc_X_v, WV, num_col, col_W, num_col, context);
  for (int i = 0; i < col_W; ++i){
    PhantomPlaintext ecd_b_v;
    vector<double> bv_vec(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
        if(bias_vec[j] == 1){
            bv_vec[j] = bV[i];
        }
    }
    encoder.encode(bv_vec, V[i].params_id(), V[i].scale(), ecd_b_v);
    //Plaintext temp = bK[i];
    evaluator.mod_switch_to_inplace(ecd_b_v, V[i].params_id());
    V[i].scale() = scale;
    ecd_b_v.scale() = scale;
    evaluator.add_plain_inplace(V[i],ecd_b_v);
  }

  gettimeofday(&tend1,NULL);
  double QKV_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
  cout <<"Compute Q, K, V time = "<<QKV_time<<". ";
  append_csv_row("../results.csv", "single_layer_without_softmaxboot(QKV)", QKV_time);

  //cout <<"Q = XW_Q+bQ, K = XW_K+bK, V = XW_V+bV. "<<endl;
  //cout <<"      Modulus chain index for Q,K: "<< seal_context.get_context_data(Q[0].parms_id())->chain_index()<<endl;
  //cout <<"      Modulus chain index for V: "<< seal_context.get_context_data(V[0].parms_id())->chain_index()<<endl;

/*
  //for test
  cout <<"Decrypt + decode result of Q: "<<endl;
    //decrypt and decode
    for (int i = 0; i < 5; ++i){
        Plaintext plain_result;
        decryptor.decrypt(Q[i], plain_result);
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
   

    //QK
    gettimeofday(&tstart1,NULL);
    vector<PhantomCiphertext> QK = ct_ct_matrix_mul_colpacking(Q, K, RotK, relin_keys,
     context, col_W, 128, col_W, 128, num_batch);
    gettimeofday(&tend1,NULL);
    double QK_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Compute QK^T time = "<<QK_time<<". ";
    append_csv_row("../results.csv", "single_layer_without_softmaxboot(QK^T)", QK_time);


    //cout <<"Q * K^T. "<<endl;
    //cout <<"    Modulus chain index for QK^T: "<< seal_context.get_context_data(QK[0].parms_id())->chain_index()<<endl;


    //for test
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < 5; ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(QK[i], plain_result);
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

    for (int i = QK.size()-5; i < QK.size(); ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(QK[i], plain_result);
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



    //softmax(QK^T)
    // gettimeofday(&tstart1,NULL);
    // vector<PhantomCiphertext> enc_softmax = softmax_boot(QK,bias_vec,input_num,context,
    //     relin_keys,iter,sk,bootstrapper_att, layer_id);
    // gettimeofday(&tend1,NULL);
    // double softmax_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    // cout <<"Compute softmax time = "<<softmax_time<<". ";

    //cout <<"    Modulus chain index for the result: "<< seal_context.get_context_data(enc_softmax[0].parms_id())->chain_index()<<endl;

/*
    //for test
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < enc_softmax.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_softmax[i], plain_result);
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
    
    //softmax(QK)V
    gettimeofday(&tstart1,NULL);

    //for (int i = 0; i < V.size(); ++i){
   //   evaluator.mod_switch_to_inplace(V[i], enc_softmax[i].parms_id());
    //}
    // vector<PhantomCiphertext> output = ct_ct_matrix_mul_diagpacking(enc_softmax, V, RotK, relin_keys, 
    //     context, 128, 128, col_W, 128, num_batch);
    for (int i = 0; i < QK.size(); ++i) {
        for (int j = 0; j < 10; ++j){
            evaluator.mod_switch_to_next_inplace(QK[i]);
        }
    }
    vector<PhantomCiphertext> output = ct_ct_matrix_mul_diagpacking(QK, V, RotK, relin_keys, 
        context, 128, 128, col_W, 128, num_batch);

    gettimeofday(&tend1,NULL);
    double softmaxV_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Compute softmax*V time = "<<softmaxV_time<<endl;
    append_csv_row("../results.csv", "single_layer_without_softmaxboot(softmax*V)", softmaxV_time);

    //cout <<"    Modulus chain index for the result: "<< seal_context.get_context_data(output[0].parms_id())->chain_index()<<endl;

    //double total_time = QKV_time+QK_time+softmax_time+softmaxV_time;
    //cout <<"Total time for single attention block = "<<total_time<<"s. "<<endl;

    return output;


}