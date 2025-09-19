#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

void BPmax_test()
{
    cout << "Task: test BPmax with column encoding in CKKS scheme: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40,
                                                                       40, 40, 40, 40, 60}));
    double scale = pow(2.0, 40);

    // parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40,40, 40,
    //     40,40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,60}));
    // double scale = pow(2.0,40);

    // SEALContext context(parms);
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
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
    // Evaluator evaluator(seal_context, encoder);
    // size_t slot_count = encoder.slot_count();
    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    struct timeval tstart1, tend1;

    // construct input
    int num_X = 256;
    int num_row = 128;
    int num_col = 64;
    cout << "Number of matrices in one batch = " << num_X << endl;
    vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));
    for (int i = 0; i < num_X; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int k = 0; k < num_col; ++k)
            {
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
    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);

    cout << "encode and encrypt X. " << endl;
    cout << "Modulus chain index for x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    vector<int> bias_vec(slot_count, 1);
    gettimeofday(&tstart1, NULL);

    int ctnum = enc_ecd_x.size();

    vector<PhantomCiphertext> enc_softmax(ctnum);

    PhantomPlaintext ecdc;
    encoder.encode(5.0, enc_ecd_x[0].params_id(), enc_ecd_x[0].scale(), ecdc);

    PhantomPlaintext ecdrd;
    encoder.encode(0.2, enc_ecd_x[0].params_id(), enc_ecd_x[0].scale(), ecdrd);

    for (int i = 0; i < ctnum; ++i)
    {
        enc_softmax[i] = enc_ecd_x[i];
        // compute (x+c)^p, c = 5, p = 5

        // enc_softmax[i] = enc_softmax[i]+c
        evaluator.add_plain_inplace(enc_softmax[i], ecdc);

        PhantomCiphertext tmp;
        // tmp = enc_softmax[i]^2
        evaluator.square(enc_softmax[i], tmp);
        evaluator.relinearize_inplace(tmp, relin_keys);
        evaluator.rescale_to_next_inplace(tmp);

        // tmp = enc_softmax[i]^4
        evaluator.square_inplace(tmp);
        evaluator.relinearize_inplace(tmp, relin_keys);
        evaluator.rescale_to_next_inplace(tmp);

        // enc_softmax[i] = tmp*enc_softmax[i]
        evaluator.mod_switch_to_inplace(enc_softmax[i], tmp.params_id());
        evaluator.multiply_inplace(enc_softmax[i], tmp);
        evaluator.relinearize_inplace(enc_softmax[i], relin_keys);
        evaluator.rescale_to_next_inplace(enc_softmax[i]);
        /*
        //enc_softmax[i] = enc_softmax[i]/Rd
        evaluator.mod_switch_to_inplace(ecdrd,enc_softmax[i].parms_id());
        evaluator.multiply_plain_inplace(enc_softmax[i]);
        evaluator.rescale_to_next_inplace(enc_softmax[i]);
        */
    }
    cudaDeviceSynchronize();
    gettimeofday(&tend1, NULL);
    double softmax_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "BPmax time = " << softmax_time << endl;
    cout << "Modulus chain index for BPmax(x): " << context.get_context_data(enc_softmax[0].params_id()).chain_depth() << endl;
}

void BatchLN_test()
{
    cout << "Task: test the BatchLN function in CKKS scheme. " << endl;
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 60}));
    double scale = pow(2.0, 40);

    // SEALContext context(parms);
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
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
    // Evaluator evaluator(seal_context, encoder);
    // size_t slot_count = encoder.slot_count();
    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    struct timeval tstart1, tend1;

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
                input_x[i][j][k] = 0.001 * (double)k;
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

    gettimeofday(&tstart1, NULL);

    int ctnum = enc_ecd_x.size();

    vector<PhantomCiphertext> enc_layernorm(ctnum);

    PhantomPlaintext ecdln;
    encoder.encode(5.0, enc_ecd_x[0].params_id(), enc_ecd_x[0].scale(), ecdln);

    // compute u=(x0+x1+...+x768)
    PhantomCiphertext ave_x = enc_ecd_x[0];
    for (int i = 1; i < ctnum; ++i)
    {
        evaluator.add_inplace(ave_x, enc_ecd_x[i]);
    }

    // compute (x-u)*ln
    for (int i = 0; i < ctnum; ++i)
    {
        enc_layernorm[i] = enc_ecd_x[i];
        evaluator.sub_inplace(enc_layernorm[i], ave_x);
        evaluator.multiply_plain_inplace(enc_layernorm[i], ecdln);
        evaluator.rescale_to_next_inplace(enc_layernorm[i]);
    }

    gettimeofday(&tend1, NULL);
    double layernorm_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "BatchLN time = " << layernorm_time << endl;

    cout << "Modulus chain index for BatchLN: " << context.get_context_data(enc_layernorm[0].params_id()).chain_depth() << endl;
}
