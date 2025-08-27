#include "include.cuh"



using namespace std;
using namespace phantom;
using namespace moai;

void exp_inv_test(){
    cout <<"Task: test exp function and inverse function in CKKS scheme: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40,40, 40,
        40,40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,60}));
    double scale = pow(2.0,40);
    long sparse_slots = 32768; 
    parms.set_sparse_slots(sparse_slots);
    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    // KeyGenerator keygen(context);
    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    // Encryptor encryptor(context, public_key);
    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);

    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    struct timeval tstart1, tend1;

    //construct input ciphertext
    double pt = 1;
    for (int ind = 0; ind < 1; ++ind){
        
    //double pt = 1.5;
    pt += 0.5;
    PhantomPlaintext one;
    encoder.encode(pt, scale, one);
    PhantomCiphertext x;
    encryptor.encrypt(one, x);
    cout <<"input encode and encrypt, pt = "<<pt<<endl;
    cout <<"Modulus chain index for x: "<< context.get_context_data(x.params_id()).chain_depth()<<endl;

    //compute exp
    PhantomCiphertext enc_exp = exp(x,context,relin_keys);
    cout <<"exp. "<<endl;
    cout <<"Modulus chain index for e^x: "<< context.get_context_data(enc_exp.params_id()).chain_depth()<<endl;
    //decrypt and decode
    PhantomPlaintext dec_exp;
    decryptor.decrypt(enc_exp,dec_exp);
    vector<double> result;
    encoder.decode(dec_exp, result);
    cout <<"Decrypt of "<<pt<<"^128: "<<endl;
    for (int ind = 0 ; ind < 5 ; ++ind){
        cout <<result[ind]<<" ";
    }
    cout <<"... ";
    for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
        cout <<result[ind]<<" ";
    }
    cout <<endl;
    cout <<endl;

    cout <<"Modulus chain index for the result: "<< context.get_context_data(x.params_id()).chain_depth()<<endl;
    //compute inverse
    PhantomCiphertext enc_inv = inverse(x,context,relin_keys,8);
    cout <<"inv. "<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(enc_inv.params_id()).chain_depth()<<endl;
    //decrypt and decode
    PhantomPlaintext dec_inv;
    decryptor.decrypt(enc_inv,dec_inv);
    //vector<double> result;
    encoder.decode(dec_inv, result);
    cout <<"Decrypt of inverse of "<<pt<<": "<<endl;
    for (int ind = 0 ; ind < 5 ; ++ind){
        cout <<result[ind]<<" ";
    }
    cout <<"... ";
    for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
        cout <<result[ind]<<" ";
    }
    cout <<endl;
}

}

void softmax_test(){
    cout <<"Task: test softmax function (without bootstrapping version) with column encoding in CKKS scheme: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {51, 46, 46,46, 46,
        46,46,46,46,46,46,46,46,46, 46,46,46,58}));
    long sparse_slots = 16384;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0,46);

   // parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40,40, 40,
   //     40,40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,60}));
   // double scale = pow(2.0,40);

    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);


    // KeyGenerator keygen(context);
    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    struct timeval tstart1, tend1;

    //construct input
    int num_X = 128;
    int num_row = 128;
    int num_col = 64;
    cout <<"Number of matrices in one batch = "<<num_X<<endl;
    vector<vector<vector<double>>> input_x(num_X,vector<vector<double>>(num_row, vector<double>(num_col,0)));
    for (int i = 0; i < num_X; ++i){
        for (int j = 0 ; j < num_row ; ++j){
            for (int k = 0 ; k < num_col ; ++k){
                input_x[i][j][k] = 0.01;
            }
            /*
            for (int k = 0; k < 10; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<"... ";
            for (int k = num_col-10 ; k<num_col ; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<endl;
            */
        }

    }
    //encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context,public_key);

    cout <<"encode and encrypt X. "<<endl;
    cout <<"Modulus chain index for x: "<< context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()<<endl;

    vector<int> bias_vec(slot_count,1);
    gettimeofday(&tstart1,NULL);

    vector<PhantomCiphertext> enc_softmax = softmax(enc_ecd_x,bias_vec, num_X, context,relin_keys,4,secret_key);

    gettimeofday(&tend1,NULL);
    double softmax_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"softmax time = "<<softmax_time<<endl;
    cout <<"Modulus chain index for softmax(x): "<< context.get_context_data(enc_softmax[0].params_id()).chain_depth()<<endl;
    append_csv_row("../results.csv", "softmax_without_boot", softmax_time);
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < 5; ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_softmax[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
    cout <<"......"<<endl;
    for (int i = num_col-5; i < num_col; ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_softmax[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
}


void softmax_boot_test(){
    cout <<"Task: test softmax function (with bootstrapping version) with column encoding in CKKS scheme: "<<endl;

    //bootstrapping parameters
    long boundary_K = 25;
  long deg = 59;
  long scale_factor = 2;
  long inverse_deg = 1;

  long logN = 15;
  long loge = 10;

  long logn = 13;
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
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, logp);
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40,40, 40,
    //    40,40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,60}));
    //double scale = pow(2.0,40);

    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    // KeyGenerator keygen(context);
    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    
    PhantomGaloisKey gal_keys_boot;

    // Encryptor encryptor(context, public_key);
    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    // Evaluator evaluator(context, encoder);
    // size_t slot_count = encoder.slot_count();

    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();
    //cout <<slot_count<<endl;

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &phantom_encoder, &relin_keys, &gal_keys_boot, scale);

    //prepare for bootstrapping
    // Bootstrapper bootstrapper(
    //   loge,
    //   logn,
    //   logN - 1,
    //   total_level,
    //   scale,
    //   boundary_K,
    //   deg,
    //   scale_factor,
    //   inverse_deg,
    //   context,
    //   keygen,
    //   encoder,
    //   encryptor,
    //   decryptor,
    //   evaluator,
    //   relin_keys,
    //   gal_keys_boot);

    
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
      &ckks_evaluator);

    cout << "preparing bootstrapping..." << endl;
    bootstrapper.prepare_mod_polynomial();

    //cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    // keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);
    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

  //cout << "Generating Linear Transformation Coefficients..." << endl;
  bootstrapper.generate_LT_coefficient_3();

    struct timeval tstart1, tend1;

    //construct input
    int num_X = 128;
    int num_row = 128;
    int num_col = 128;
    cout <<"Number of matrices in one batch = "<<num_X<<endl;
    vector<vector<vector<double>>> input_x(num_X,vector<vector<double>>(num_row, vector<double>(num_col,0)));
    for (int i = 0; i < num_X; ++i){
        for (int j = 0 ; j < num_row ; ++j){
            for (int k = 0 ; k < num_col ; ++k){
                input_x[i][j][k] = 0.01;
            }
            /*
            for (int k = 0; k < 10; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<"... ";
            for (int k = num_col-10 ; k<num_col ; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<endl;
            */
        }

    }
    //encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context,public_key);

    cout <<"encode and encrypt X. "<<endl;
    cout <<"Modulus chain index for x: "<< context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()<<endl;

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
        evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
    }

    cout <<"Modulus chain index before attention block: "<< context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()<<endl;

    vector<int> bias_vec(slot_count,1);
    gettimeofday(&tstart1,NULL);

    vector<PhantomCiphertext> enc_softmax = softmax_boot(enc_ecd_x,bias_vec,11,context,
        relin_keys,16,secret_key,bootstrapper, 10);

    gettimeofday(&tend1,NULL);
    double softmax_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"softmax time = "<<softmax_time<<endl;
    cout <<"Modulus chain index for softmax(x): "<< context.get_context_data(enc_softmax[0].params_id()).chain_depth()<<endl;
    append_csv_row("../results.csv", "softmax_with_boot", softmax_time);
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < 5; ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_softmax[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
    cout <<"......"<<endl;
    for (int i = num_col-5; i < num_col; ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_softmax[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
}






