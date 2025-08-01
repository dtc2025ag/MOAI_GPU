#include <chrono>
using namespace chrono;
#include "Bootstrapper.h"
#include "ckks_evaluator.h"

using namespace std;
using namespace seal;

const int num_X = 256;
const int num_row = 128;
const int num_col = 768;

int num_input = 5;

vector<vector<vector<double>>> input_x0(num_X,vector<vector<double>>(num_row, vector<double>(num_col,0)));
vector<vector<vector<double>>> input_x(num_X,vector<vector<double>>(num_row, vector<double>(num_col,0)));
vector<double> layernorm1_gamma(num_col,0.0);
vector<double> layernorm1_beta(num_col,0.0);

void read_input_layernorm(){
    ifstream fin;

    fin.open("first_layernorm_output.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file selfoutput_linear.txt"<<endl;
    }
    char a;
    //the test file has 5 input vectors, length of each vector = 768
    
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

    //ifstream fin;
    fin.open("selfoutput_linear.txt");
    if(!fin.is_open()){
        cout <<"Cannot open file selfoutput_linear.txt"<<endl;
    }
    char a;
    //the test file has 5 input vectors, length of each vector = 768
    
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

    //read layernorm2 weight
    fin.open("layer_"+to_string(1)+"/Attention/SelfOutput/parms/self_output_LayerNorm_weight.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_LayerNorm_weight.csv"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>layernorm1_gamma[k];
    }
    fin.close();
    cout <<"LayerNorm1 last element: "<<layernorm1_gamma[num_col-1]<<endl;

    //read layernorm2 bias
    fin.open("layer_"+to_string(1)+"/Attention/SelfOutput/parms/self_output_LayerNorm_bias.csv");
    if(!fin.is_open()){
        cout <<"Cannot open file self_output_LayerNorm_bias.txt"<<endl;
    }
    for(int k = 0 ; k < num_col ; ++k){
        fin>>layernorm1_beta[k];
    }
    fin.close();
    cout <<"LayerNorm1 bias last element: "<<layernorm1_beta[num_col-1]<<endl;
}

void layernorm_bootstrapping_test(){
    read_input_layernorm();
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
  int remaining_level = 20;
  int boot_level = 14;  // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
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

    cout <<"Task: test the layernorm function + bootstrapping in CKKS scheme. "<<endl;
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = (size_t)(1 << logN);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    double scale = pow(2.0, logp);

    SEALContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;

    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    Evaluator evaluator(context, encoder);
    size_t slot_count = encoder.slot_count();
    cout <<"slot count = "<<slot_count<<endl;

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
      gal_keys);

  cout << "Generating Optimal Minimax Polynomials..." << endl;
  bootstrapper.prepare_mod_polynomial();

  cout << "Adding Bootstrapping Keys..." << endl;
  vector<int> gal_steps_vector;
  gal_steps_vector.push_back(0);
  for (int i = 0; i < logN - 1; i++) {
    gal_steps_vector.push_back((1 << i));
  }
  bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

  keygen.create_galois_keys(gal_steps_vector, gal_keys);

  bootstrapper.slot_vec.push_back(logn);

  cout << "Generating Linear Transformation Coefficients..." << endl;
  bootstrapper.generate_LT_coefficient_3();

    struct timeval tstart1, tend1;  

    //encode + encrypt
    vector<Ciphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context,public_key);
    vector<int> input_len(num_X,0);
    input_len[0] = 5;
    vector <int> b_vec = bias_vec(input_len,num_X,num_row);

    cout <<"encode and encrypt X. "<<endl;
    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;


    //mod switch to remaining level
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }
    cout <<"Modulus chain index before layernorm: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    //encrypt embedding
    vector<Ciphertext> enc_ecd_x0 = batch_input(input_x0, num_X, num_row, num_col, scale, context,public_key);
    //mod switch to remaining level
    //encrypt input + encrypt embedding
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        evaluator.mod_switch_to_inplace(enc_ecd_x0[i],enc_ecd_x[i].parms_id());
        evaluator.add_inplace(enc_ecd_x[i],enc_ecd_x0[i]);
    }
    cout <<"Modulus chain index before layernorm: "<< context.get_context_data(enc_ecd_x0[0].parms_id())->chain_index()<<endl;

    

    //plaintext result
    vector<double> sum(num_input,0);
    cout <<"Sum of x: ";
    for (int i = 0; i < num_input; ++i){
        for (int j = 0 ; j < num_col ; ++j){
            input_x[0][i][j] += input_x0[0][i][j];
            sum[i] += input_x[0][i][j];
        }
    }
    for (int i = 0; i < num_input; ++i){
        cout <<sum[i]<<" ";
    }
    cout <<endl;

    cout <<"Var: ";
    vector<double>var(num_input,0);
    for (int i = 0; i < num_input; ++i){
        for (int j = 0 ; j < num_col ; ++j){
            var[i] += (768.0*input_x[0][i][j]-sum[i])*(768.0*input_x[0][i][j]-sum[i]);
        }
        var[i] /= (768.0*768.0);
    }
    for (int i = 0; i < num_input; ++i){
        cout <<var[i]<<" ";
    }
    cout <<endl;

    cout <<"Inv sqrt var: ";
    vector<double> inv_sqrt_var(num_input,0);
    for (int i = 0; i < num_input; ++i){
        inv_sqrt_var[i] = 1.0/(sqrt(var[i]));
    }
    for (int i = 0; i < num_input; ++i){
        cout <<inv_sqrt_var[i]<<" ";
    }
    cout <<endl;

    vector<vector<double>> layernorm_pt(num_input,vector<double>(num_col,0.0));
    for (int i = 0; i < num_input; ++i){
        for (int j = 0; j < num_col; ++j){
            layernorm_pt[i][j] = (layernorm1_gamma[j]/sqrt(768.0))*(768.0*input_x[0][i][j]-sum[i])*inv_sqrt_var[i]+layernorm1_beta[j];
            //layernorm_pt[i][j] = (layernorm2_gamma[j]/768.0)*(768.0*input_x[0][i][j]-sum[i])*inv_sqrt_var[i]+layernorm2_beta[j];
        }
    }
    for (int i = 0; i < 5; ++i){
        for(int j = 0 ; j < num_input ; ++j){
            cout <<layernorm_pt[j][i]<<" ";
        }
        cout <<endl;
    }


    
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> output = layernorm2(enc_ecd_x,layernorm1_gamma,layernorm1_beta,b_vec,context,relin_keys,secret_key);

    gettimeofday(&tend1,NULL);
    double layernorm_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"layernorm time = "<<layernorm_time<<endl;

    //size_t layer_num = context.get_context_data(output[0].parms_id())->chain_index();
    cout <<"Modulus chain index after layernorm: "<< context.get_context_data(output[0].parms_id())->chain_index()<<endl;

    //delete enc_ecd_x
    vector<Ciphertext>().swap(enc_ecd_x);

    //decrypt
    

    cout <<"Decrypt + decode result of layernorm: "<<endl;
    for (int i = 0; i < output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
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

    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < num_col; ++i){
        while(context.get_context_data(output[i].parms_id())->chain_index() != 0){
        evaluator.mod_switch_to_next_inplace(output[i]);
        }
    }
    
    cout <<"Modulus chain index before bootstrapping: "<<context.get_context_data(output[1].parms_id())->chain_index();

    //bootstrap 1 ciphertext
    vector<Ciphertext> rtn(num_col);

    gettimeofday(&tstart1,NULL);

    omp_set_num_threads(56);

    #pragma omp parallel for

    for(int i = 0 ; i < 96 ; ++i){
        for(int j = 0 ; j < 8 ; ++j){
            bootstrapper.bootstrap_3(rtn[i*8+j],output[i*8+j]);
        }
    }
    //bootstrapper.bootstrap_3(rtn, output[1]);

    gettimeofday(&tend1,NULL);
    double boot_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time<<endl;
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;

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

