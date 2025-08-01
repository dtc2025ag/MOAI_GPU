using namespace std;
using namespace seal;

vector<Ciphertext> ct_pt_matrix_mul_wo_pre(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W, int col_X, int col_W, int row_W, 
  const SEALContext& seal_context){

  vector<Ciphertext> output(col_W);
  double scale = enc_X[0].scale();

  if(col_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

    #pragma omp parallel for 
    for (int i = 0; i < col_W; ++i){
        //encode w[0][i]
        Plaintext ecd_w_0_i;
        encoder.encode(W[0][i], enc_X[0].parms_id(), enc_X[0].scale(), ecd_w_0_i);
        //enc_X[0]*ecd_w[0][i]
        evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i]);
        //evaluator.rescale_to_next_inplace(output[i]);

        for (int j = 1 ; j < row_W ; ++j){
          //encode w[j][i]
          Plaintext ecd_w_j_i;
          encoder.encode(W[j][i], enc_X[j].parms_id(), enc_X[j].scale(), ecd_w_j_i);

          //enc_X[j]*ecd_w[j][i]
          Ciphertext temp;
          evaluator.multiply_plain(enc_X[j], ecd_w_j_i, temp);
          //evaluator.rescale_to_next_inplace(temp);
          evaluator.add_inplace(output[i],temp);
        }

        evaluator.rescale_to_next_inplace(output[i]);
        output[i].scale()=scale;
    }


  

  return output;

}

vector<Ciphertext> ct_pt_matrix_mul_wo_pre_large(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W, int col_X, int col_W, int row_W, 
  const SEALContext& seal_context){

  vector<Ciphertext> output(col_W);
  double scale = enc_X[0].scale();

  if(col_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

    int col_W_t = col_W/128;

    #pragma omp parallel for 


    for (int i = 0; i < 128; ++i){
      for (int k = 0 ; k < col_W_t ; ++k){
        //encode w[0][i]
        Plaintext ecd_w_0_i;
        encoder.encode(W[0][i*col_W_t+k], enc_X[0].parms_id(), enc_X[0].scale(), ecd_w_0_i);
        //enc_X[0]*ecd_w[0][i]
        evaluator.multiply_plain(enc_X[0], ecd_w_0_i, output[i*col_W_t+k]);
        //evaluator.rescale_to_next_inplace(output[i]);

        for (int j = 1 ; j < row_W ; ++j){
          //encode w[j][i]
          Plaintext ecd_w_j_i;
          encoder.encode(W[j][i*col_W_t+k], enc_X[j].parms_id(), enc_X[j].scale(), ecd_w_j_i);

          //enc_X[j]*ecd_w[j][i]
          Ciphertext temp;
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

vector<Ciphertext> ct_pt_matrix_mul_wo_pre_w_mask(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W,const vector<int> & bias_vec, int col_X, int col_W, int row_W, 
  const SEALContext& seal_context){

  vector<Ciphertext> output(col_W);
  double scale = enc_X[0].scale();

  if(col_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  size_t slot_count = encoder.slot_count();

  int col_W_t = col_W/128;
  //cout <<col_W_t<<endl;

  #pragma omp parallel for 

  for (int i = 0; i < 128; ++i){
    for (int k = 0 ; k < col_W_t ; ++k){
      //encode w[0][i]
      vector<double> temp(slot_count,0);
      for (int j = 0 ; j < slot_count ; ++j){
        if(bias_vec[j] == 1){
          temp[j] = W[0][i*col_W_t+k];
        }
      }
      Plaintext ecd_w_0_i;
      encoder.encode(temp, enc_X[0].parms_id(), enc_X[0].scale(), ecd_w_0_i);
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
        Plaintext ecd_w_j_i;
        encoder.encode(tempw, enc_X[j].parms_id(), enc_X[j].scale(), ecd_w_j_i);

        //enc_X[j]*ecd_w[j][i]
        Ciphertext tempx;
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

vector<Ciphertext> ct_pt_matrix_mul(const vector<Ciphertext> & enc_X, 
  const vector<vector<Plaintext>> & W, int col_X, int col_W, int row_W, 
  const SEALContext& seal_context){

  vector<Ciphertext> output(col_W);

  if(col_X != row_W){
    cout <<"ERROR: bad dimensions of X or W. "<<endl;
    return output;
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  #pragma omp parallel for 

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
      Ciphertext temp;
      evaluator.multiply_plain(enc_X[j], W[j][i], temp);
      //evaluator.rescale_to_next_inplace(temp);
      evaluator.add_inplace(output[i],temp);
    }

    evaluator.rescale_to_next_inplace(output[i]);

  }

  return output;

}