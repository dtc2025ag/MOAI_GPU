#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

void ct_pt_matrix_mul_test()
{
    cout << "Task: test Ct-Pt matrix multiplication in CKKS scheme: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                 {60, 40, 40, 60}));
    //{60, 40, 40, 60}));
    long sparse_slots = 16384;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    // PhantomKeyGenerator keygen(context);
    // PhantomSecretKey secret_key = keygen.secret_key();
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // PhantomDecryptor decryptor(context, secret_key);
    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Encryptor encryptor(&context, &public_key);
    Evaluator evaluator(&context, &phantom_encoder);

    size_t slot_count = encoder.slot_count();

    // struct timeval tstart1, tend1;

    // construct input
    int num_X = 128;
    int num_row = 128;
    int num_col = 768;
    cout << "Number of matrices in one batch = " << num_X << endl;
    vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));
    for (int i = 0; i < num_X; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int k = 0; k < num_col; ++k)
            {
                input_x[i][j][k] = (double)j + 1.0;
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
    cout << "Matrix X size = " << num_row << " * " << num_col << endl;

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);

    cout << "encode and encrypt X. " << endl;
    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    /*
    //decrypt


    for (int i = 0; i < num_col; ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<"decrypt + decode result of "<<i+1<<"-th ciphertext: "<<endl;
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
    */

    // construct W
    int col_W = 64;
    vector<vector<double>> W(num_col, vector<double>(col_W, 1.0 / 128.0));
    cout << "Matrix W size = " << num_col << " * " << col_W << endl;

    /*
    //encode W
    vector<vector<Plaintext>> ecd_w(num_col,vector<Plaintext>(col_W));
    for (int i = 0; i < num_col; ++i){
        for (int j = 0 ; j < col_W ; ++j){
            encoder.encode(W[i][j], scale, ecd_w[i][j]);
        }
    }
    cout <<"encode W. "<<endl;
    */
    cout << "Encrypted col-packing X * ecd W = Encrypted col-packing XW. " << endl;

    // matrix multiplication
    //  gettimeofday(&tstart1,NULL);
    std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();
    cudaDeviceSynchronize();
    vector<PhantomCiphertext> ct_pt_mul = ct_pt_matrix_mul_wo_pre(enc_ecd_x, W, num_col, col_W, num_col, context);

    // gettimeofday(&tend1,NULL);
    // double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    // cout <<"Ct-Pt matrix multiplication time = "<<ct_pt_matrix_mul_time<<endl;
    std::chrono::_V2::system_clock::time_point end = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cudaDeviceSynchronize();
    cout << "Ct-Pt matrix multiplication time = " << duration.count() << " ms" << endl;
    cout << "Modulus chain index for the result: " << context.get_context_data(ct_pt_mul[0].params_id()).chain_depth() << endl;
    append_csv_row("../results.csv", "ct_pt_matrix_mul_without_preprocessing", duration.count() / 1000.0);
    cout << "Decrypt + decode result: " << endl;
    // decrypt and decode
    for (int i = 0; i < 32; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
    cout << "......" << endl;
    for (int i = col_W - 32; i < col_W; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
}

void ct_pt_matrix_mul_w_preprocess_test()
{
    cout << "Task: test Ct-Pt matrix multiplication with preprocess in CKKS scheme: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                 {60, 40, 40, 60}));
    //{60, 40, 40, 60}));
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    // KeyGenerator keygen(context);
    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);

    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    // size_t slot_count = encoder.slot_count();

    // struct timeval tstart1, tend1;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    // PhantomDecryptor decryptor(context, secret_key);
    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Encryptor encryptor(&context, &public_key);
    Evaluator evaluator(&context, &phantom_encoder);

    size_t slot_count = encoder.slot_count();

    // construct input
    int num_X = 256;
    int num_row = 128;
    int num_col = 768;
    cout << "Number of matrices in one batch = " << num_X << endl;
    vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));
    for (int i = 0; i < num_X; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int k = 0; k < num_col; ++k)
            {
                input_x[i][j][k] = (double)j + 1.0;
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
    cout << "Matrix X size = " << num_row << " * " << num_col << endl;

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);

    cout << "encode and encrypt X. " << endl;
    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    /*
    //decrypt


    for (int i = 0; i < num_col; ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<"decrypt + decode result of "<<i+1<<"-th ciphertext: "<<endl;
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
    */

    // construct W
    int col_W = 64;
    vector<vector<double>> W(num_col, vector<double>(col_W, 1.0 / 128.0));
    cout << "Matrix W size = " << num_col << " * " << col_W << endl;

    // encode W
    vector<vector<PhantomPlaintext>> ecd_w(num_col, vector<PhantomPlaintext>(col_W));

    // #pragma omp parallel for
    std::chrono::_V2::system_clock::time_point start_encode = high_resolution_clock::now();
    // cudaDeviceSynchronize();

    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < col_W; ++j)
        {
            encoder.encode(W[i][j], scale, ecd_w[i][j]);
        }
    }
    cout << "encode W. " << endl;

    cout << "Encrypted col-packing X * ecd W = Encrypted col-packing XW. " << endl;
    std::chrono::_V2::system_clock::time_point end_encode = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_encode = end_encode - start_encode;
    cudaDeviceSynchronize();
    cout << "[DEBUG]pre-encoding time: " << duration_encode.count() << " ms" << endl;
    // matrix multiplication
    // gettimeofday(&tstart1,NULL);
    std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();

    vector<PhantomCiphertext> ct_pt_mul = ct_pt_matrix_mul(enc_ecd_x, ecd_w, num_col, col_W, num_col, context);

    // gettimeofday(&tend1,NULL);
    // double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    // cout <<"Ct-Pt matrix multiplication time (pre process not included) = "<<ct_pt_matrix_mul_time<<endl;
    std::chrono::_V2::system_clock::time_point end = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cudaDeviceSynchronize();
    cout << "Ct-Pt matrix multiplication time (pre process not included) = " << duration.count() << " ms" << endl;
    cout << "Modulus chain index for the result: " << context.get_context_data(ct_pt_mul[0].params_id()).chain_depth() << endl;

    cout << "Decrypt + decode result: " << endl;
    // decrypt and decode
    for (int i = 0; i < 5; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
    cout << "......" << endl;
    for (int i = col_W - 5; i < col_W; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th ciphertext: ";
        for (int ind = 0; ind < 5; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 5; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }
}