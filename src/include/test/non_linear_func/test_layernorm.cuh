#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

void sqrt_inv_test()
{
    cout << "Task: test inverse and sqrt function in CKKS scheme: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40,
                                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
    long sparse_slots = 16384;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    cout << "context.total_chain_index(): " << context.get_context_data(0).chain_depth() << endl;

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

    // construct input ciphertext
    double pt = 768;
    PhantomPlaintext one;
    encoder.encode(pt, scale, one);
    PhantomCiphertext x;
    encryptor.encrypt(one, x);
    cout << "input encode and encrypt" << endl;
    cout << "Modulus chain index for x: " << context.get_context_data(x.params_id()).chain_depth() << endl;

    // compute invert sqrt
    PhantomCiphertext enc_sqrt_inv = invert_sqrt(x, 4, 2, context, relin_keys);
    cout << "invert sqrt. " << endl;
    cout << "Modulus chain index for invert sqrt: " << context.get_context_data(enc_sqrt_inv.params_id()).chain_depth() << endl;
    // decrypt and decode
    PhantomPlaintext dec_inv;
    decryptor.decrypt(enc_sqrt_inv, dec_inv);
    vector<double> result;
    encoder.decode(dec_inv, result);
    cout << "Decrypt of 1/sqrt(" << pt << "): " << endl;
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
    cout << endl;
}

void layernorm_test()
{
    cout << "Task: test the layernorm function in CKKS scheme. " << endl;
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40,
                                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);
    // cout << "context.total_chain_index(): " << context.get_context_data(0).chain_depth() << endl;
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
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);

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

    for (int i = 0; i < 5; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th input: ";
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
    for (int i = num_col - 5; i < num_col; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << i + 1 << "-th input: ";
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

    vector<double> gamma(num_col, 0.5);
    vector<double> beta(num_col, 2.0);
    vector<int> bias_vec(slot_count, 1);

    double var = 0.0;
    for (int i = 0; i < num_col; ++i)
    {
        double temp = 0.001 * (double)i - 0.3835;
        temp *= temp;
        var += temp;
    }
    var /= num_col;
    cout << "var = " << var << endl;
    vector<double> ln(num_col, 0.0);
    for (int i = 0; i < 10; ++i)
    {
        ln[i] = (gamma[i] * (0.001 * (double)i - 0.3835) * (1 / sqrt(var))) + beta[i];
        cout << ln[i] << " ";
    }
    cout << endl;
    for (int i = num_col - 10; i < num_col; ++i)
    {
        ln[i] = (gamma[i] * (0.001 * (double)i - 0.3835) * (1 / sqrt(var))) + beta[i];
        cout << ln[i] << " ";
    }
    cout << endl;

    gettimeofday(&tstart1, NULL);

    vector<PhantomCiphertext> output = layernorm(enc_ecd_x, gamma, beta, bias_vec, context, relin_keys, secret_key);

    gettimeofday(&tend1, NULL);
    double layernorm_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "layernorm time = " << layernorm_time << endl;

    cout << "Modulus chain index for layernorm: " << context.get_context_data(output[0].params_id()).chain_depth() << endl;
    append_csv_row("../results.csv", "layernorm", layernorm_time);
    // decrypt

    for (int i = 0; i < 10; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
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
    for (int i = num_col - 10; i < num_col; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
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
