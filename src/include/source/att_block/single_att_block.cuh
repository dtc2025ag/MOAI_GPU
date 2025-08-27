#include <chrono>
using namespace chrono;
#include "Bootstrapper.h"
#include "ckks_evaluator.h"

using namespace std;
using namespace seal;


vector<Ciphertext> single_att_block(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & WQ, const vector<vector<double>> & WK,
  const vector<vector<double>> & WV, const vector<double> &bQ, 
  const vector<double> &bK, const vector<double> &bV, const vector<int> &bias_vec, int input_num,
  const SEALContext& seal_context, const RelinKeys &relin_keys, const GaloisKeys & RotK, Bootstrapper &bootstrapper_att,
  int num_batch, const SecretKey & sk, int iter, int layer_id){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  size_t slot_count = encoder.slot_count();
  //for test
  Decryptor decryptor(seal_context, sk);
  double scale = enc_X[0].scale();

  int col_W = WQ[0].size();
  int num_col = enc_X.size();
  //cout <<"number of column of x = "<<num_col<<", number of column of WQ = "<<col_W<<endl;
  struct timeval tstart1, tend1;

  gettimeofday(&tstart1,NULL);
  vector<Ciphertext> Q = ct_pt_matrix_mul_wo_pre(enc_X, WQ, num_col, col_W, num_col, seal_context);
  for (int i = 0; i < col_W; ++i){
    //cout <<i<<" ";
    Plaintext ecd_b_q;
    vector<double> bq_vec(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
        if(bias_vec[j] == 1){
            bq_vec[j] = bQ[i];
        }
    }
    encoder.encode(bq_vec, Q[i].parms_id(), Q[i].scale(), ecd_b_q);
    evaluator.mod_switch_to_inplace(ecd_b_q, Q[i].parms_id());
    Q[i].scale() = scale;
    ecd_b_q.scale() = scale;
    evaluator.add_plain_inplace(Q[i],ecd_b_q);
  }


  //cout <<"Q += b"<<endl;
  vector<Ciphertext> K = ct_pt_matrix_mul_wo_pre(enc_X, WK, num_col, col_W, num_col, seal_context);
  
  for (int i = 0; i < col_W; ++i){
    Plaintext ecd_b_k;
    vector<double> bk_vec(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
        if(bias_vec[j] == 1){
            bk_vec[j] = bK[i];
        }
    }
    encoder.encode(bk_vec, K[i].parms_id(), K[i].scale(), ecd_b_k);
    evaluator.mod_switch_to_inplace(ecd_b_k, K[i].parms_id());
    K[i].scale() = scale;
    ecd_b_k.scale() = scale;
    evaluator.add_plain_inplace(K[i],ecd_b_k);

  }

  vector<Ciphertext> enc_X_v(num_col);

  #pragma omp parallel for

  for (int i = 0; i < num_col; ++i){
      enc_X_v[i] = enc_X[i];
      while (seal_context.get_context_data(enc_X_v[i].parms_id())->chain_index()>3){
        evaluator.mod_switch_to_next_inplace(enc_X_v[i]);
      }
  }
  vector<Ciphertext> V = ct_pt_matrix_mul_wo_pre(enc_X_v, WV, num_col, col_W, num_col, seal_context);
  for (int i = 0; i < col_W; ++i){
    Plaintext ecd_b_v;
    vector<double> bv_vec(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
        if(bias_vec[j] == 1){
            bv_vec[j] = bV[i];
        }
    }
    encoder.encode(bv_vec, V[i].parms_id(), V[i].scale(), ecd_b_v);
    evaluator.mod_switch_to_inplace(ecd_b_v, V[i].parms_id());
    V[i].scale() = scale;
    ecd_b_v.scale() = scale;
    evaluator.add_plain_inplace(V[i],ecd_b_v);
  }

  gettimeofday(&tend1,NULL);
  double QKV_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
  cout <<"Compute Q, K, V time = "<<QKV_time<<". ";

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
    vector<Ciphertext> QK = ct_ct_matrix_mul_colpacking(Q, K, RotK, relin_keys,
     seal_context, col_W, 128, col_W, 128, num_batch);
    gettimeofday(&tend1,NULL);
    double QK_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Compute QK^T time = "<<QK_time<<". ";

/*
    //for test
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < 5; ++i){
        Plaintext plain_result;
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
        Plaintext plain_result;
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
*/


    //softmax(QK^T)
    gettimeofday(&tstart1,NULL);
    vector<Ciphertext> enc_softmax = softmax_boot(QK,bias_vec,input_num,seal_context,
        relin_keys,iter,sk,bootstrapper_att, layer_id);
    gettimeofday(&tend1,NULL);
    double softmax_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Compute softmax time = "<<softmax_time<<". ";

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
    vector<Ciphertext> output = ct_ct_matrix_mul_diagpacking(enc_softmax, V, RotK, relin_keys, 
        seal_context, 128, 128, col_W, 128, num_batch);

    gettimeofday(&tend1,NULL);
    double softmaxV_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Compute softmax*V time = "<<softmaxV_time<<endl;

    return output;


}