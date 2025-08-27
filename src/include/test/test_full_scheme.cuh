#include <chrono>
using namespace chrono;
#include "Bootstrapper.h"
#include "ckks_evaluator.h"
using namespace std;
using namespace seal;

const int num_X = 256;
const int num_row = 128;
const int num_col = 768;
const int num_inter = 3072;
const double sqrt_d = 8.0;
int num_input = 5;

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
vector<double> layernorm1_gamma(num_col,0.0);
vector<double> layernorm1_beta(num_col,0.0);

vector<vector<double>> inter_weight(num_col, vector<double>(num_inter,0.0));
vector<double> inter_bias(num_inter,0.0);
vector<vector<double>> final_weight(num_inter,vector<double>(num_col,0.0));
vector<double> final_bias(num_col,0.0);
vector<double> layernorm2_gamma(num_col,0.0);
vector<double> layernorm2_beta(num_col,0.0);


void read_input(){
    ifstream fin;
    fin.open("layer_0/Attention/BertSelfAttention/allresults/embedded_inputs.csv");
    //fin.open("layer_0_output.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file embedded_inputs.csv"<<endl;
    }

    string lineStr;
    int ind_num_input = 0;
    while(getline(fin, lineStr)){
        stringstream ss(lineStr);
        string str;
        vector<string> lineArray;
        while(getline(ss,str,',')){
            lineArray.push_back(str);
        }
        for (int i = 0; i < num_col; ++i){
            input_x[0][ind_num_input][i] = stod(lineArray[i]);
        }
        ind_num_input++;

    }
    fin.close();
    //for test
    cout <<input_x[0][4][0]<<" "<<input_x[0][4][num_col-1]<<endl;
}

void read_input_2(){
    double sum = 0.0;
    ifstream fin;
    fin.open("layer_6_output.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file layer_6_output.txt"<<endl;
    }
    char a;
    //the test file has 11 input vectors, length of each vector = 768
    
    for (int i = 0; i < num_input; ++i){
        for (int j = 0 ; j < num_col-1 ; ++j){
            fin >>input_x[0][i][j];
            //fin >>a;
            sum += input_x[0][i][j];
        }
        fin >>input_x[0][i][num_col-1];
        sum += input_x[0][i][num_col-1];
    }
    fin.close();
    //for test
    cout <<sum<<endl;
    cout <<input_x[0][4][0]<<" "<<input_x[0][4][num_col-1]<<endl;
}

void read_weights(int layer_id){
    ifstream fin;
    //read matrix Q, size of Q = 12*64*768
    fin.open("layer_"+to_string(layer_id)+"/Attention/BertSelfAttention/parms/query_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file query_weight.csv"<<endl;
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
    cout <<"WQ last element: "<<WQ[num_head-1][num_col-1][col_W-1]<<endl;

    //Q = Q/sqrt(d')
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < num_col; ++i){
            for(int j = 0 ; j < col_W ; ++j){
                WQ[k][i][j] = WQ[k][i][j]/sqrt_d;
            }
        }
    }

    //read matrix K
    fin.open("layer_"+to_string(layer_id)+"/Attention/BertSelfAttention/parms/key_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file key_weight.csv"<<endl;
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
    cout <<"WK last element: "<<WK[num_head-1][num_col-1][col_W-1]<<endl;

    //read matrix V
    fin.open("layer_"+to_string(layer_id)+"/Attention/BertSelfAttention/parms/value_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file value_weight.csv"<<endl;
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
    cout <<"WV last element: "<<WV[num_head-1][num_col-1][col_W-1]<<endl;

    //read self output weight
    fin.open("layer_"+to_string(layer_id)+"/Attention/SelfOutput/parms/self_output_dense_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_dense_weight.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        for (int i = 0; i < num_col-1; ++i){
            fin >> selfoutput[i][k];
            fin >>a;
        }
        fin>>selfoutput[num_col-1][k];
    }
    fin.close();
    cout <<"selfoutput last element: "<<selfoutput[num_col-1][num_col-1]<<endl;

    //read layernorm1 weight
    fin.open("layer_"+to_string(layer_id)+"/Attention/SelfOutput/parms/self_output_LayerNorm_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_LayerNorm_weight.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>layernorm1_gamma[k];
    }
    fin.close();
    cout <<"LayerNorm1 last element: "<<layernorm1_gamma[num_col-1]<<endl;
}

void read_bias(int layer_id){
    ifstream fin;
    //read bias Q, size of Q = 64
    fin.open("layer_"+to_string(layer_id)+"/Attention/BertSelfAttention/parms/query_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file query_bias.csv"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            fin>>bQ[k][i];
        }
    }
    fin.close();
    //for test
    cout <<"Q bias last element: "<<bQ[num_head-1][col_W-1]<<endl;

    //bias Q = bias Q / sqrt_d'
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            bQ[k][i] = bQ[k][i] / sqrt_d;
        }
    }

    fin.open("layer_"+to_string(layer_id)+"/Attention/BertSelfAttention/parms/key_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file key_bias.csv"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            fin>>bK[k][i];
        }
    }
    fin.close();
    //for test
    cout <<"K bias last element: "<<bK[num_head-1][col_W-1]<<endl;

    fin.open("layer_"+to_string(layer_id)+"/Attention/BertSelfAttention/parms/value_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file value_bias.csv"<<endl;
    }
    for(int k = 0 ; k < num_head ; ++k){
        for (int i = 0; i < col_W; ++i){
            fin>>bV[k][i];
        }
    }
    fin.close();
    //for test
    cout <<"v bias last element: "<<bV[num_head-1][col_W-1]<<endl;

    //read self output bias
    fin.open("layer_"+to_string(layer_id)+"/Attention/SelfOutput/parms/self_output_dense_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_dense_bias.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>selfoutput_bias[k];
    }
    fin.close();
    //for test
    cout <<"selfoutput bias last element: "<<selfoutput_bias[num_col-1]<<endl;

    //read layernorm1 weight
    fin.open("layer_"+to_string(layer_id)+"/Attention/SelfOutput/parms/self_output_LayerNorm_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_LayerNorm_bias.txt"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>layernorm1_beta[k];
    }
    fin.close();
    cout <<"LayerNorm1 bias last element: "<<layernorm1_beta[num_col-1]<<endl;
}

void read_feed_forward_param(int layer_id){
    ifstream fin;
    char a;
    //read inter weight
    fin.open("layer_"+to_string(layer_id)+"/Intermediate/parms/intermediate_dense_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file intermediate_dense_weight.csv"<<endl;
    }
    for(int k = 0 ; k < num_inter ; ++k){
        for (int i = 0; i < num_col-1; ++i){
            fin >> inter_weight[i][k];
            fin >>a;
        }
        fin>>inter_weight[num_col-1][k];
    }
    fin.close();
    cout <<"inter_weight last element: "<<inter_weight[num_col-1][num_inter-1]<<endl;

    //read inter bias
    fin.open("layer_"+to_string(layer_id)+"/Intermediate/parms/intermediate_dense_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file intermediate_dense_bias.csv"<<endl;
    }
    for(int k = 0 ; k < num_inter ; ++k){
        fin >> inter_bias[k];
    }
    fin.close();
    cout <<"inter_bias last element: "<<inter_bias[num_inter-1]<<endl;

    //read final weight
    fin.open("layer_"+to_string(layer_id)+"/Output/parms/final_output_dense_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file final_output_dense_weight.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        for (int i = 0; i < num_inter-1; ++i){
            fin >> final_weight[i][k];
            fin >>a;
        }
        fin>>final_weight[num_inter-1][k];
    }
    fin.close();
    cout <<"final_weight last element: "<<final_weight[num_inter-1][num_col-1]<<endl;

    //read final bias
    fin.open("layer_"+to_string(layer_id)+"/Output/parms/final_output_dense_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file final_output_dense_bias.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin >> final_bias[k];
    }
    fin.close();
    cout <<"final_bias last element: "<<final_bias[num_col-1]<<endl;

    //read layernorm2 weight
    fin.open("layer_"+to_string(layer_id)+"/Output/parms/final_output_LayerNorm_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file final_output_LayerNorm_weight.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>layernorm2_gamma[k];
    }
    fin.close();
    cout <<"LayerNorm2 weights last element: "<<layernorm2_gamma[num_col-1]<<endl;

    //read layernorm2 bias
    fin.open("layer_"+to_string(layer_id)+"/Output/parms/final_output_LayerNorm_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file final_output_LayerNorm_bias.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>layernorm2_beta[k];
    }
    fin.close();
    cout <<"LayerNorm2 bias last element: "<<layernorm2_beta[num_col-1]<<endl;
}

void all_layer_test(){
    cout <<"Task: test BERT in CKKS scheme: "<<endl;

    read_input();
 
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
  int remaining_level_att = 15;
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
    //cout <<slot_count<<endl;


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

    cout << "preparing bootstrapping..." << endl;
    bootstrapper.prepare_mod_polynomial();

    //cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);

    bootstrapper.slot_vec.push_back(logn);

  //cout << "Generating Linear Transformation Coefficients..." << endl;
  bootstrapper.generate_LT_coefficient_3();


    struct timeval tstart1, tend1;
 
    //encode + encrypt
    vector<Ciphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context,public_key);
    vector<int> input_len(num_X,0);
    input_len[0] = num_input;
    vector <int> b_vec = bias_vec(input_len,num_X,num_row);
    //cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<endl;
    //cout <<"Modulus chain index for x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    vector<vector<vector<double>>>().swap(input_x);

    vector<Ciphertext> enc_ecd_x_copy(num_col);
    for (int i = 0; i < num_col; ++i){
        enc_ecd_x_copy[i] = enc_ecd_x[i];
    }

    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x_copy[i]);
        }
    }

    #pragma omp parallel for 

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }

for (int layer_id = 0; layer_id < 12; ++layer_id){
    cout <<endl;
    cout <<"---------------Layer No. "<<layer_id<<"-------------------"<<endl;

    read_weights(layer_id);
    read_bias(layer_id);
    read_feed_forward_param(layer_id);
    cout <<"Read input, weights, bias from txt files. "<<endl;

    //mod switch to remaining level
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < (remaining_level- remaining_level_att); ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }

    //mod switch to next level
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
    }

    cout <<"Modulus chain index before attention block: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;
/*
    for (int j = 0; j < 5; ++j){
            Plaintext plain_result;
            decryptor.decrypt(enc_ecd_x[j], plain_result);
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
*/

    vector<vector<Ciphertext>> att_block(num_head);

    gettimeofday(&tstart1,NULL);

    for (int i = 0; i < 12; ++i){
        att_block[i] = single_att_block(enc_ecd_x, WQ[i], WK[i], WV[i], bQ[i], bK[i], bV[i], b_vec, num_input,
        context, relin_keys, gal_keys, bootstrapper, num_X, secret_key,16,layer_id);
    /*    
        cout <<"Decrypt + decode result of ";
        cout <<i+1<<"-th head: "<<endl;
        for (int j = 0; j < 5; ++j){
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
    */
    }
    

    gettimeofday(&tend1,NULL);
    double att_block_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Attention block time = "<<att_block_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(att_block[2][0].parms_id())->chain_index()<<endl;

    /*
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int k = 0; k < num_head; ++k){
        //cout <<k+1<<"-th head: "<<endl;
        for (int i = 0; i < 2; ++i){
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
   

    cout <<endl;
    */


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
    vector<Ciphertext> att_selfoutput = ct_pt_matrix_mul_wo_pre_w_mask(att_output, selfoutput, b_vec, num_col, num_col, num_col, context);
    int att_selfoutput_size = att_selfoutput.size();
    //cout <<"num of ct in att_selfoutput = "<<att_selfoutput_size<<endl;
    for (int i = 0; i < num_col; ++i){
        Plaintext ecd_self_bias;
        vector<double> self_bias_vec(slot_count,0);
        for (int j = 0; j < slot_count; ++j){
            if(b_vec[j] == 1){
                self_bias_vec[j] = selfoutput_bias[i];
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

/*
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
*/
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < att_selfoutput_size; ++i){
        while(context.get_context_data(att_selfoutput[i].parms_id())->chain_index() != 0){
        evaluator.mod_switch_to_next_inplace(att_selfoutput[i]);
        }
    }

    vector<Ciphertext> rtn(att_selfoutput_size);

    //cout<<"bootstrapping start. "<<endl;

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

    //for (int i = 0; i < rtn.size(); ++i){
    //    evaluator.mod_switch_to_next_inplace(rtn[i]);
    //}
    //cout <<"Modulus chain index before layernorm: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;



    vector<Ciphertext>().swap(att_selfoutput);

    
    
    
    //LayerNorm
    //cout <<"LayerNorm start. "<<endl;
    gettimeofday(&tstart1,NULL);

    //rtn+enc_ecd_x_copy
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_x_copy[i], rtn[i].parms_id());
        evaluator.add_inplace(rtn[i],enc_ecd_x_copy[i]);
    }

/*
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
*/
    vector<Ciphertext> layernorm_selfoutput = layernorm(rtn,layernorm1_gamma,layernorm1_beta, b_vec,
        context,relin_keys,secret_key);

    gettimeofday(&tend1,NULL);
    double layernorm_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"layernorm time = "<<layernorm_time<<endl;
    cout <<"Modulus chain index after layernorm: "<< context.get_context_data(layernorm_selfoutput[0].parms_id())->chain_index()<<endl;
    //vector<Ciphertext>().swap(enc_ecd_x_copy);
/*
    cout <<"Decrypt + decode result of layernorm: "<<endl;
    for (int i = 0; i < layernorm_selfoutput.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(layernorm_selfoutput[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        int iscout = 0;
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.001){
                if(iscout == 0){
                    cout <<"( "<<ind<<", "<<result[ind]<<"). ";
                    iscout ++;
                }
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    //bootstrapping
    int layernorm_selfoutput_size = layernorm_selfoutput.size();
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < layernorm_selfoutput_size; ++i){
        while(context.get_context_data(layernorm_selfoutput[i].parms_id())->chain_index() != 0){
        evaluator.mod_switch_to_next_inplace(layernorm_selfoutput[i]);
        }
    }

    vector<Ciphertext> boot_layer(layernorm_selfoutput_size);

    //cout<<"bootstrapping start. "<<endl;

    gettimeofday(&tstart1,NULL);

    #pragma omp parallel for

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn[i*6+j],layernorm_selfoutput[i*6+j]);
            boot_layer[i*6+j] = rtn[i*6+j];
        }
    }

    #pragma omp parallel for

    for (int i = 0; i < rtn.size(); ++i) {
        for (int j = 0; j < 11; ++j){
            evaluator.mod_switch_to_next_inplace(rtn[i]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time2<<endl;
    //cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(boot_layer[0].parms_id())->chain_index()<<endl;

    vector<Ciphertext>().swap(layernorm_selfoutput);

/*
    //decrypt and decode
    cout <<"Decrypt + decode result of bootstrapping: "<<endl;
    for (int i = 0; i < 10; ++i){
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
*/
    //rtn * inter_weight + inter_bias

    cout <<"Modulus chain index before intermediate linear: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> inter_output = ct_pt_matrix_mul_wo_pre_large(rtn, inter_weight, num_col, num_inter, num_col, context);
    int inter_output_size = inter_output.size();
    //cout <<"num of ct in inter_output = "<<inter_output_size<<endl;
    /*
    cout <<"scale of inter_output = "<<log2(inter_output[0].scale())<<endl;
    cout <<"Decrypt + decode result of intermediate_linear wo bias: "<<endl;
    for (int i = 0; i < inter_output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(inter_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.001){
                cout <<"( "<<ind<<", "<<result[ind]<<"). ";
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    for (int i = 0; i < num_inter; ++i){
        Plaintext ecd_inter_bias;
        vector<double> inter_bias_vec(slot_count,0);
        for (int j = 0; j < slot_count; ++j){
            if(b_vec[j] == 1){
                inter_bias_vec[j] = inter_bias[i];
            }
        }
        encoder.encode(inter_bias_vec, inter_output[i].parms_id(), inter_output[i].scale(), ecd_inter_bias);
        evaluator.mod_switch_to_inplace(ecd_inter_bias, inter_output[i].parms_id());
        inter_output[i].scale() = scale;
        ecd_inter_bias.scale() = scale;
        evaluator.add_plain_inplace(inter_output[i],ecd_inter_bias);

    }

    gettimeofday(&tend1,NULL);
    double inter_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Inter layer time = "<<inter_time<<endl;
    cout <<"Modulus chain index after inter layer: "<< context.get_context_data(inter_output[0].parms_id())->chain_index()<<endl;

/*
    cout <<"Decrypt + decode result of intermediate_linear: "<<endl;
    for (int i = 0; i < inter_output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(inter_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        int iscout = 0;
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.001){
                if(iscout == 0){
                    cout <<"( "<<ind<<", "<<result[ind]<<"). ";
                    iscout ++;
                }
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    vector<Ciphertext> gelu_output(num_inter);

    gettimeofday(&tstart1,NULL);

    #pragma omp parallel for

    for (int i = 0; i < 96; ++i){
        for (int j = 0 ; j < 32; ++j){
            gelu_output[i*32+j] = gelu_v2(inter_output[i*32+j],context,relin_keys,secret_key);
        }
    }
    

    gettimeofday(&tend1,NULL);
    double gelu_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"gelu time = "<<gelu_time<<endl;

    cout <<"Modulus chain index for gelu: "<< context.get_context_data(gelu_output[0].parms_id())->chain_index()<<endl;

    vector<Ciphertext>().swap(inter_output);
/*
    cout <<"Decrypt + decode result of intermediate_gelu: "<<endl;
    for (int i = 0; i < 2; ++i){
        Plaintext plain_result;
        decryptor.decrypt(gelu_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        int iscout = 0;
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            
        //    else if(result[ind] >= 0.001){
        //        if(iscout == 0){
        //            cout <<"( "<<ind<<", "<<result[ind]<<"). ";
        //            iscout ++;
        //        }
        //    }
            
        }
        cout <<endl;
    }

    cout <<endl;
*/
    //gelu * final_weight + final_bias
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> final_output = ct_pt_matrix_mul_wo_pre_w_mask(gelu_output, final_weight,b_vec, num_inter, num_col, num_inter, context);
    int final_output_size = final_output.size();
    //cout <<"num of ct in final_output = "<<final_output_size<<endl;
    for (int i = 0; i < num_col; ++i){
        Plaintext ecd_final_bias;
        vector<double> final_bias_vec(slot_count,0);
        for (int j = 0; j < slot_count; ++j){
            if(b_vec[j] == 1){
                final_bias_vec[j] = final_bias[i];
            }
        }
        encoder.encode(final_bias_vec, final_output[i].parms_id(), final_output[i].scale(), ecd_final_bias);
        evaluator.mod_switch_to_inplace(ecd_final_bias, final_output[i].parms_id());
        final_output[i].scale() = scale;
        ecd_final_bias.scale() = scale;
        evaluator.add_plain_inplace(final_output[i],ecd_final_bias);

    }

    gettimeofday(&tend1,NULL);
    double final_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Final layer time = "<<final_time<<endl;
    cout <<"Modulus chain index after final layer: "<< context.get_context_data(final_output[0].parms_id())->chain_index()<<endl;
/*
    cout <<"Decrypt + decode result of intermediate_final: "<<endl;
    for (int i = 0; i < final_output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(final_output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        int iscout = 0;
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.001){
                if(iscout == 0){
                    cout <<"( "<<ind<<", "<<result[ind]<<"). ";
                    iscout ++;
                }
            }
        }
        cout <<endl;
    }

    cout <<endl;
*/
    //bootstrapping
    //int final_output_size = final_output.size();
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < final_output_size; ++i){
        while(context.get_context_data(final_output[i].parms_id())->chain_index() != 0){
        evaluator.mod_switch_to_next_inplace(final_output[i]);
        }
    }

    //cout<<"bootstrapping start. "<<endl;
    vector<Ciphertext> rtn2(768);
    gettimeofday(&tstart1,NULL);

    #pragma omp parallel for

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn2[i*6+j],final_output[i*6+j]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time3 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time3<<endl;
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn2[0].parms_id())->chain_index()<<endl;

   // for (int i = 0; i < rtn2.size(); ++i){
    //    evaluator.mod_switch_to_next_inplace(rtn2[i]);
   // }
    //cout <<"Modulus chain index before layernorm: "<< context.get_context_data(rtn2[0].parms_id())->chain_index()<<endl;

    vector<Ciphertext>().swap(final_output);

    //cout <<"LayerNorm start. "<<endl;
    gettimeofday(&tstart1,NULL);


    //rtn+enc_ecd_x_copy
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(boot_layer[i], rtn2[i].parms_id());
        evaluator.add_inplace(rtn2[i],boot_layer[i]);
    }
/*
    cout <<"Decrypt + decode result before layernorm: "<<endl;
    for (int i = 0; i < 5; ++i){
        Plaintext plain_result;
        decryptor.decrypt(rtn2[i], plain_result);
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
*/
    vector<Ciphertext> layernorm_finaloutput = layernorm2(rtn2,layernorm2_gamma,layernorm2_beta,b_vec,
        context,relin_keys,secret_key);

    gettimeofday(&tend1,NULL);
    double layernorm_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"layernorm time = "<<layernorm_time2<<endl;
    cout <<"Modulus chain index after layernorm: "<< context.get_context_data(layernorm_finaloutput[0].parms_id())->chain_index()<<endl;
    vector<Ciphertext>().swap(rtn);

    cout <<"Decrypt + decode result of one layer: "<<endl;
    ofstream fout;
    fout.open("layer_"+to_string(layer_id)+".txt");
    for (int i = 0; i < layernorm_finaloutput.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(layernorm_finaloutput[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        fout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                fout <<result[ind]<<", ";
            }
        }
        fout <<endl;
    }

    fout <<endl;
    fout.close();

    //bootstrapping
    int layernorm2_size = layernorm_finaloutput.size();
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < layernorm2_size; ++i){
        while(context.get_context_data(layernorm_finaloutput[i].parms_id())->chain_index() != 0){
        evaluator.mod_switch_to_next_inplace(layernorm_finaloutput[i]);
        }
    }

    cout<<"bootstrapping start. "<<endl;
    gettimeofday(&tstart1,NULL);

    #pragma omp parallel for

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn2[i*6+j],layernorm_finaloutput[i*6+j]);
            enc_ecd_x[i*6+j] = rtn2[i*6+j];
            enc_ecd_x_copy[i*6+j] = rtn2[i*6+j];
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time4 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time4<<endl;
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn2[0].parms_id())->chain_index()<<endl;

    vector<Ciphertext>().swap(layernorm_finaloutput);
/*
    cout <<"Decrypt + decode result of one layer: "<<endl;
    for (int i = 0; i < rtn2.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(rtn2[i], plain_result);
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
*/
    double total_time = att_block_time+selfoutput_time+layernorm_time+inter_time+gelu_time+final_time+layernorm_time2
    +boot_time+boot_time2+boot_time3+boot_time4;
    cout <<"Total time for one layer: "<<total_time<<", amortized time: "<<total_time/256.0<<endl;

 //   cout <<enc_ecd_x.size()<<endl;
 //   cout <<enc_ecd_x_copy.size()<<endl;

}

   
}








