#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

void gelu_test()
{
    cout << "Task: test the GeLU v2 function in CKKS scheme. " << endl;
    // bootstrapping parameters
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
    int boot_level = 0; // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2

    int remaining_level = 10;
    int total_level = remaining_level + boot_level;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++)
    {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < boot_level; i++)
    {
        coeff_bit_vec.push_back(logq);
    }
    coeff_bit_vec.push_back(log_special_prime);

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = (size_t)(1 << logN);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    // parms.set_sparse_slots(sparse_slots);
    //  long sparse_slots = (1 << logn);
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, logp);
    // parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40,40, 40,
    //     40,40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,60}));
    // double scale = pow(2.0,40);

    // PhantomContext context(parms, true, sec_level_type::none);
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

    // GaloisKeys gal_keys;
    // keygen.create_galois_keys(gal_keys);
    // GaloisKeys gal_keys_boot;

    // Encryptor encryptor(context, public_key);
    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    // Evaluator evaluator(context, encoder);

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
                input_x[i][j][k] = (double)(i - 128) / 15.0;
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

    cout << "encode and encrypt X. num of ct = " << enc_ecd_x.size() << endl;
    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < boot_level; ++j)
        {
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }

    cout << "Modulus chain index for enc x: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;
    /*
        vector<double> pt_input(num_col,0);
        for (int i = 0; i < num_col; ++i){
            pt_input[i] = -13;
        }
        cout <<"Gelu in pt: ";
        vector<double> pt_output = gelu_plain(pt_input);
        for (int i = 0; i < num_col; ++i){
            cout <<pt_output[i]<<" ";
        }
        cout <<endl;
    */

    const int max_threads = omp_get_max_threads();
    const int nthreads = std::max(1, std::min(max_threads, 102));

    if (stream_pool.size() < static_cast<size_t>(nthreads))
    {
        stream_pool.reserve(nthreads);
        for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
        {
            stream_pool.emplace_back();
        }
    }
    if (nthreads == 1)
    {
        stream_pool[0] = *phantom::util::global_variables::default_stream;
    }

    vector<PhantomCiphertext> output(num_col);

    gettimeofday(&tstart1, NULL);

    //   omp_set_num_threads(56);

    //   #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto &stream = stream_pool[tid];
#pragma omp for schedule(static)
        for (int i = 0; i < num_col; ++i)
        {
            output[i] = gelu_v2(enc_ecd_x[i], context, relin_keys, secret_key, stream);
        }
        cudaStreamSynchronize(stream.get_stream());
    }

    gettimeofday(&tend1, NULL);
    double gelu_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "gelu time = " << gelu_time << endl;
    cout << "gelu time (amortized) = " << gelu_time / num_col << endl;
    cout << "Modulus chain index for gelu: " << context.get_context_data(output[0].params_id()).chain_depth() << endl;
    append_csv_row("../results.csv", "gelu_v2", gelu_time);
    // decrypt
    // cout << "Decrypt + decode result of intermediate_gelu: " << endl;
    // for (int i = 0; i < output.size(); ++i)
    // {
    //     PhantomPlaintext plain_result;
    //     decryptor.decrypt(output[i], plain_result);
    //     vector<double> result;
    //     encoder.decode(plain_result, result);
    //     cout << i + 1 << "-th ciphertext: ";
    //     for (int ind = 0; ind < 10; ++ind)
    //     {
    //         cout << result[ind] << ", ";
    //     }
    //     cout << endl;
    // }

    cout << endl;
}
