#include <chrono>
using namespace chrono;
#include "Bootstrapper.h"
#include "ckks_evaluator.h"
using namespace std;
using namespace seal;

const int num_X = 256;
const int num_row = 128;
const int num_col = 768;
const double sqrt_d = 8.0;
int num_input = 11;

vector<vector<vector<double>>> input_x(num_X,vector<vector<double>>(num_row, vector<double>(num_col,0)));

//paras for attention block 
int col_W = 64;
int num_head = 12;
vector<vector<vector<double>>> WQ(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));
vector<vector<vector<double>>> WK(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));
vector<vector<vector<double>>> WV(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));

vector<vector<double>> bQ(num_head,vector<double>(col_W,0.0));
vector<vector<double>> bK(num_head,vector<double>(col_W,0.0));
vector<vector<double>> bV(num_head,vector<double>(col_W,0.0));

vector<vector<double>> selfoutput(num_col, vector<double>(num_col,0.0));
vector<double> selfoutput_bias(num_col,0.0);

void read_input(){
    ifstream fin;
    fin.open("att_block_weights/embedded_inputs.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file embedded_inputs.txt"<<endl;
    }
    char a;
    //the test file has 11 input vectors, length of each vector = 768
    
    for (int i = 0; i < num_input; ++i){
        for (int j = 0 ; j < num_col-1 ; ++j){
            fin >>input_x[0][i][j];
            fin >>a;
        }
        fin >>input_x[0][i][num_col-1];
    }
    fin.close();
    //for test
    //cout <<input_x[0][10][0]<<" "<<input_x[0][10][num_col-1]<<endl;
}

void read_weights(){
    ifstream fin;
    //read matrix Q, size of Q = 12*64*768
    fin.open("att_block_weights/query_weight.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file query_weight.txt"<<endl;
    }
    char a;
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            for (int j = 0 ; j < num_col-1 ; ++j){
                fin >> WQ[k][j][i];
                fin >>a;
            }
            fin>>WQ[k][num_col-1][i];
        }
    }
    
    fin.close();
    //for test
    cout <<WQ[num_head-1][num_col-1][col_W-1]<<endl;

    //Q = Q/sqrt(d')
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < num_col; ++i){
            for(int j = 0 ; j < col_W ; ++j){
                WQ[k][i][j] = WQ[k][i][j]/sqrt_d;
            }
        }
    }

    //read matrix K
    fin.open("att_block_weights/key_weight.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file key_weight.txt"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            for (int j = 0 ; j < num_col-1 ; ++j){
                fin >> WK[k][j][i];
                fin >>a;
            }
            fin>>WK[k][num_col-1][i];
        }
    }
    fin.close();
    //for test
    cout <<WK[num_head-1][num_col-1][col_W-1]<<endl;

    //read matrix V
    fin.open("att_block_weights/value_weight.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file value_weight.txt"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            for (int j = 0 ; j < num_col-1 ; ++j){
                fin >> WV[k][j][i];
                fin >>a;
            }
            fin>>WV[k][num_col-1][i];
        }
    }
    fin.close();
    //for test
    cout <<WV[num_head-1][num_col-1][col_W-1]<<endl;

    //read self output weight
    fin.open("self_output_weights/self_output_dense_weight.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_dense_weight.txt"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        for (int i = 0; i < num_col-1; ++i){
            fin >> selfoutput[i][k];
            fin >>a;
        }
        fin>>selfoutput[num_col-1][k];
    }
    fin.close();
    cout <<selfoutput[num_col-1][num_col-1]<<endl;
}

void read_bias(){
    ifstream fin;
    //read bias Q, size of Q = 64
    fin.open("att_block_weights/query_bias.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file query_bias.txt"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            fin>>bQ[k][i];
        }
    }
    fin.close();
    //for test
    cout <<bQ[num_head-1][col_W-1]<<endl;

    //bias Q = bias Q / sqrt_d'
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            bQ[k][i] = bQ[k][i] / sqrt_d;
        }
    }

    fin.open("att_block_weights/key_bias.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file key_bias.txt"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            fin>>bK[k][i];
        }
    }
    fin.close();
    //for test
    cout <<bK[num_head-1][col_W-1]<<endl;

    fin.open("att_block_weights/value_bias.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file value_bias.txt"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            fin>>bV[k][i];
        }
    }
    fin.close();
    //for test
    cout <<bV[num_head-1][col_W-1]<<endl;

    //read self output bias
    fin.open("self_output_weights/self_output_dense_bias.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_dense_bias.txt"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>selfoutput_bias[k];
    }
    fin.close();
    //for test
    cout <<selfoutput_bias[num_col-1]<<endl;
}

void multi_att_block_test(){
    cout <<"Task: test attention block with 12 heads of BERT in CKKS scheme: "<<endl;

    read_input();
    read_weights();
    read_bias();
    cout <<"Read input, weights, bias from txt files. "<<endl;

    //bootstrapping parameters
    long boundary_K = 25;
  long deg = 59;
  long scale_factor = 2;
  long inverse_deg = 1;

  long logN = 16;
  long loge = 10;

  long logn = 15;
  long sparse_slots = (1 << logn);

  int logp = 46;
  int logq = 51;
  int log_special_prime = 58;

  int secret_key_hamming_weight = 192;

  // Calculation required
  int remaining_level_att = 14;
  int boot_level = 14;  // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
  int total_level_att = remaining_level_att + boot_level;

  int remaining_level = 20;
  int total_level = remaining_level + boot_level;

  vector<int> coeff_bit_vec;
  coeff_bit_vec.push_back(logq);
  for (int i = 0; i < remaining_level; i++) {
    coeff_bit_vec.push_back(logp);
  }
  for (int i = 0; i < boot_level; i++) {
    coeff_bit_vec.push_back(logq);
  }
  coeff_bit_vec.push_back(log_special_prime);


    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = (size_t)(1 << logN);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    double scale = pow(2.0, logp);

    SEALContext context(parms, true, sec_level_type::none);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    GaloisKeys gal_keys_boot;

    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    Evaluator evaluator(context, encoder);
    size_t slot_count = encoder.slot_count();
    cout <<slot_count<<endl;


    //prepare for bootstrapping
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
      context,
      keygen,
      encoder,
      encryptor,
      decryptor,
      evaluator,
      relin_keys,
      gal_keys_boot);

    cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();

    cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);

    bootstrapper.slot_vec.push_back(logn);

  cout << "Generating Linear Transformation Coefficients..." << endl;
  bootstrapper.generate_LT_coefficient_3();


    struct timeval tstart1, tend1;
 
    //encode + encrypt
    vector<Ciphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context,public_key);
    vector<int> input_len(num_X,0);
    input_len[0] = 11;
    vector <int> b_vec = bias_vec(input_len,num_X,num_row);
    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<endl;
    cout <<"Modulus chain index for x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    vector<vector<vector<double>>>().swap(input_x);

    //mod switch to remaining level
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level+(remaining_level- remaining_level_att); ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }

    //mod switch to next level
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
    }

    cout <<"Modulus chain index before attention block: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    vector<vector<Ciphertext>> att_block(num_head);

    gettimeofday(&tstart1,NULL);

    for (int i = 0; i < 12; ++i){
        att_block[i] = single_att_block(enc_ecd_x, WQ[i], WK[i], WV[i], bQ[i], bK[i], bV[i], b_vec, num_input,
        context, relin_keys, gal_keys, bootstrapper, num_X, secret_key,10);

        cout <<"Decrypt + decode result of ";
        cout <<i+1<<"-th head: "<<endl;
        for (int j = 0; j < att_block[i].size(); ++j){
            Plaintext plain_result;
            decryptor.decrypt(att_block[i][j], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            cout <<j+1<<"-th ciphertext: ";
            for (int ind = 0 ; ind < slot_count ; ++ind){
                if(b_vec[ind] == 1){
                    cout <<result[ind]<<" ";
                }
            }
            cout <<endl;
        }
    
    }
    

    gettimeofday(&tend1,NULL);
    double att_block_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Attention block time = "<<att_block_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(att_block[0][0].parms_id())->chain_index()<<endl;
    cout <<"---------------------------------------------------"<<endl;

/*
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int k = 0; k < num_head; ++k){
        cout <<k+1<<"-th head: "<<endl;
        for (int i = 0; i < att_block[k].size(); ++i){
            Plaintext plain_result;
            decryptor.decrypt(att_block[k][i], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            cout <<i+1<<"-th ciphertext: ";
            for (int ind = 0 ; ind < slot_count ; ++ind){
                if(b_vec[ind] == 1){
                    cout <<result[ind]<<", ";
                }
            }
            cout <<endl;
        }
    }
 */   

    cout <<endl;
   

    //delete enc_ecd_x
    vector<Ciphertext>().swap(enc_ecd_x);

    gettimeofday(&tstart1,NULL);

    int output_size = att_block[0].size();


    vector<Ciphertext> att_output(num_head*output_size);

    for (int i = 0; i < num_head; ++i){
        for (int j = 0 ; j < output_size ; ++j){
            att_output[i*output_size+j] = att_block[i][j];
        }
    }

    cout <<"Concatenation. size of output of attention block = "<<num_head<<" * "<<output_size<<" = "<<att_output.size()<<endl;

    vector<vector<Ciphertext>>().swap(att_block);

    //att_output * selfoutput + selfoutput_bias
    vector<Ciphertext> att_selfoutput = ct_pt_matrix_mul_wo_pre(att_output, selfoutput, num_col, num_col, num_col, context);
    int att_selfoutput_size = att_selfoutput.size();
    cout <<"num of ct in att_selfoutput = "<<att_selfoutput_size<<endl;
    for (int i = 0; i < num_col; ++i){
        Plaintext ecd_self_bias;
        vector<double> self_bias_vec(slot_count,selfoutput_bias[i]);
        for (int j = 0; j < slot_count; ++j){
            if(bias_vec[j] == 0){
                self_bias_vec[j] = 0;
            }
        }
        encoder.encode(self_bias_vec, att_selfoutput[i].parms_id(), att_selfoutput[i].scale(), ecd_self_bias);
        evaluator.mod_switch_to_inplace(ecd_self_bias, att_selfoutput[i].parms_id());
        att_selfoutput[i].scale() = scale;
        ecd_self_bias.scale() = scale;
        evaluator.add_plain_inplace(att_selfoutput[i],ecd_self_bias);
    }

    gettimeofday(&tend1,NULL);
    double selfoutput_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"selfoutput time = "<<selfoutput_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(att_selfoutput[0].parms_id())->chain_index()<<endl;

    cout <<"Decrypt + decode result of selfoutput: "<<endl;
    //decrypt and decode
    for (int k = 0; k < att_selfoutput.size(); ++k){
        cout <<k+1<<"-th ciphertext: ";
        Plaintext plain_result;
        decryptor.decrypt(att_selfoutput[k], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
        }
        cout <<endl;
    }



    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < att_selfoutput_size; ++i){
        while(context.get_context_data(att_selfoutput[i].parms_id())->chain_index() != 0){
        evaluator.mod_switch_to_next_inplace(att_selfoutput[i]);
        }
    }

    vector<Ciphertext> rtn(att_selfoutput_size);

    cout<<"bootstrapping start. "<<endl;

    gettimeofday(&tstart1,NULL);

    #pragma omp parallel for

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn[i*6+j],att_selfoutput[i*6+j]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time<<endl;
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;

    vector<Ciphertext>().swap(att_selfoutput);

    //decrypt and decode
    cout <<"Decrypt + decode result of bootstrapping: "<<endl;
    for (int i = 0; i < rtn.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(rtn[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
        }
        cout <<endl;
    }

    cout <<endl;


   
}








