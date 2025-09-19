#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

void ct_ct_matrix_mul_test()
{
    cout << "Task: test column packing Ct-Ct matrix multiplication in CKKS scheme: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                 //{60, 40, 40,40, 40,40,40,40,40,40,40,40,40,40, 40,40,40,40,60}));
                                                 {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
    long sparse_slots = 32768;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);
    // GaloisKeys gal_keys;
    // keygen.create_galois_keys(gal_keys);

    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    // Evaluator evaluator(context, encoder);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);

    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);

    size_t slot_count = encoder.slot_count();
    cout << slot_count << endl;

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

                // if(j == k){
                input_x[i][j][k] = 1.0;
                //}
            }
        }
    }

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);

    cout << "Matrix X size = " << num_row << " * " << num_col << endl;
    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    // decrypt

    for (int i = 0; i < num_col; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout << "decrypt + decode result of " << i + 1 << "-th ciphertext: " << endl;
        for (int ind = 0; ind < 10; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << "... ";
        for (int ind = slot_count - 10; ind < slot_count; ++ind)
        {
            cout << result[ind] << " ";
        }
        cout << endl;
    }

    // construct W
    vector<vector<vector<double>>> input_w(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));
    for (int i = 0; i < num_X; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int k = 0; k < num_col; ++k)
            {
                input_w[i][j][k] = ((double)j + 1.0) * 0.01;
                // input_w[i][j][k] = 1.0;
            }
        }
    }

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_w = batch_input(input_w, num_X, num_row, num_col, scale, context, public_key);

    cout << "Matrix W size = " << num_row << " * " << num_col << endl;
    cout << "Modulus chain index for enc w: " << context.get_context_data(enc_ecd_w[0].params_id()).chain_depth() << endl;

    // matrix multiplication
    cout << "Encrypted col-packing X * (Encrypted col-packing W)^T = Encrypted diag-packing XW^T " << endl;
    gettimeofday(&tstart1, NULL);

    vector<PhantomCiphertext> ct_ct_mul = ct_ct_matrix_mul_colpacking(enc_ecd_x, enc_ecd_w, gal_keys, relin_keys,
                                                                      context, num_col, num_row, num_col, num_row, num_X);

    gettimeofday(&tend1, NULL);
    double ct_ct_matrix_mul_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "column packing Ct-Ct matrix multiplication time = " << ct_ct_matrix_mul_time << endl;
    cout << "Modulus chain index for the result: " << context.get_context_data(ct_ct_mul[0].params_id()).chain_depth() << endl;
    append_csv_row("../results.csv", "ct_ct_matrix_mul_colpacking", ct_ct_matrix_mul_time);
    cout << "Decrypt + decode result: " << endl;
    // decrypt and decode
    for (int i = 0; i < 64; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_ct_mul[i], plain_result);
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
    for (int i = num_row - 64; i < num_row; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_ct_mul[i], plain_result);
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

    cout << endl;
    cout << "Task: test diag-packing Ct-Ct matrix multiplication in CKKS scheme: " << endl;

    // parms_id_type last_parms_id = ct_ct_mul[0].parms_id();
    size_t last_parms_id = ct_ct_mul[0].chain_index();

    for (int i = 0; i < enc_ecd_w.size(); ++i)
    {
        evaluator.mod_switch_to_inplace(enc_ecd_w[i], last_parms_id);
    }

    cout << "scale of encrypted w = " << log2(enc_ecd_w[0].scale());
    cout << ", scale of encrypted xw^T = " << log2(ct_ct_mul[0].scale()) << endl;
    cout << "Modulus chain index for encrypted w: " << context.get_context_data(enc_ecd_w[0].params_id()).chain_depth() << endl;
    cout << "Modulus chain index for encrypted xw^T: " << context.get_context_data(ct_ct_mul[0].params_id()).chain_depth() << endl;

    cout << "Encrypted diag-packing XW^T * Encrypted col-packing W = Encrypted diag-packing (XW^T)W. " << endl;

    gettimeofday(&tstart1, NULL);

    vector<PhantomCiphertext> ct_ct_mul_2 = ct_ct_matrix_mul_diagpacking(ct_ct_mul, enc_ecd_w, gal_keys, relin_keys,
                                                                         context, num_row, num_row, num_col, num_row, num_X);

    gettimeofday(&tend1, NULL);
    double ct_ct_matrix_mul_time2 = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "diag packing Ct-Ct matrix multiplication time = " << ct_ct_matrix_mul_time2 << endl;
    cout << "Modulus chain index for the result: " << context.get_context_data(ct_ct_mul_2[0].params_id()).chain_depth() << endl;
    append_csv_row("../results.csv", "ct_ct_matrix_mul_diagpacking", ct_ct_matrix_mul_time2);
    cout << "Decrypt + decode result: " << endl;
    // decrypt and decode
    for (int i = 0; i < 5; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_ct_mul_2[i], plain_result);
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
    for (int i = num_col - 5; i < num_col; ++i)
    {
        PhantomPlaintext plain_result;
        decryptor.decrypt(ct_ct_mul_2[i], plain_result);
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
