using namespace std;
using namespace seal;


//Interlaced batching, column-encoded. 
//In BERT, num_X = 128, num_row = 128, num_col = 768
//Output: 768 Ciphertexts
vector<Ciphertext> batch_input(const vector<vector<vector<double>>> & X, 
  int num_X, int num_row, int num_col, double scale,
  const SEALContext& seal_context,const PublicKey & pk){

  CKKSEncoder encoder(seal_context);
  Encryptor encryptor(seal_context, pk);
  size_t slot_count = encoder.slot_count();
  //cout <<"slot count = "<<slot_count<<endl;

  vector<Ciphertext> output(num_col);
  
  #pragma omp parallel for

  for (int i = 0; i < num_col; ++i){

    vector<double> vec(slot_count,0);
    for (int j = 0 ; j < num_X ; ++j){
      for (int k = 0 ; k < num_row ; ++k){
        vec[num_X*k+j] = X[j][k][i];
      }
    }

    Plaintext ecd_vec;
    encoder.encode(vec, scale, ecd_vec);
    encryptor.encrypt(ecd_vec, output[i]);

  }

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