#include "include.cuh"
#include "hf_rope_constants_seq128_dim128.cuh"
using namespace std;
using namespace phantom;
using namespace moai;

void rotary_pos_embed_test()
{
    cout << "Task: test rotary position embedding in CKKS scheme: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
                                                 {60, 40, 60}));
    //{60, 40, 40, 60}));
    long sparse_slots = 32768;
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


    vector<vector<double>> Q(128, vector<double>(32768, 1.0));
    vector<vector<double>> K(128, vector<double>(32768, 1.0));

    vector<PhantomCiphertext> enc_Q(128);
    vector<PhantomCiphertext> enc_K(128);

    for (size_t i = 0; i < 128; i++) {
        PhantomPlaintext plain_Q, plain_K;
        encoder.encode(Q[i], scale, plain_Q);
        encoder.encode(K[i], scale, plain_K);
        encryptor.encrypt(plain_Q, enc_Q[i]);
        encryptor.encrypt(plain_K, enc_K[i]);
    }
    cout << "Q, K size: " << enc_Q.size() << ", " << enc_K.size() << endl;

    using namespace hf_rope_128x128;
    std::vector<std::vector<double>> cos, sin;
    constexpr std::size_t S = 128, D = 128;
    cos.reserve(128);
    sin.reserve(128);
    for (std::size_t i = 0; i < S; ++i) {
        sin.emplace_back(SIN[i], SIN[i] + D);
        cos.emplace_back(COS[i], COS[i] + D);
    }
    cudaDeviceSynchronize();
    std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();
    auto [enc_Q_rot, enc_K_rot] = apply_rotary_pos_emb(context, enc_Q, enc_K, cos, sin);
    std::chrono::_V2::system_clock::time_point end = high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cudaDeviceSynchronize();
    cout << "RoPE time = " << duration.count() << " ms" << endl;

    // decrypt and decode
    // Q, K is column-packing , each ciphertext contains one whole column
    vector<vector<double>> res_Q(128, vector<double>(32768));
    vector<vector<double>> res_K(128, vector<double>(32768));
    for (size_t i = 0; i < 128; i++) {
        PhantomPlaintext plain_Q, plain_K;
        decryptor.decrypt(enc_Q_rot[i], plain_Q);
        decryptor.decrypt(enc_K_rot[i], plain_K);
        encoder.decode(plain_Q, res_Q[i]);
        encoder.decode(plain_K, res_K[i]);
    }
    // verify the result
    std::vector<std::vector<double>> real_Q(D, std::vector<double>(S));
    std::vector<std::vector<double>> real_K(D, std::vector<double>(S));
    // convert from [sequence, dim] to [dim, sequence]
    for (std::size_t s = 0; s < S; ++s) { 
        for (std::size_t d = 0; d < D; ++d) {
            real_Q[d][s] = Q_OUT[s][d];  
            real_K[d][s] = K_OUT[s][d];
        }
    }
    double error_Q = 0.0, error_K = 0.0;
    double max_error_Q = 0.0, max_error_K = 0.0;
    for (size_t i = 0; i < 128; i++) {
        for (size_t j = 0; j < 128; j++) {
            error_Q += abs(res_Q[i][j] - real_Q[i][j]);
            error_K += abs(res_K[i][j] - real_K[i][j]);
            max_error_Q = max(max_error_Q, abs(res_Q[i][j] - real_Q[i][j]));
            max_error_K = max(max_error_K, abs(res_K[i][j] - real_K[i][j]));
        }
    }
    cout << "The error of Q is: " << error_Q / (128 * 128) << endl;
    cout << "The error of K is: " << error_K / (128 * 128) << endl;
    cout << "The max error of Q is: " << max_error_Q << endl;
    cout << "The max error of K is: " << max_error_K << endl;
    cout << "First 10 elements of Q[0]: " << endl;
    for (size_t i = 0; i < 10; i++) {
        cout << res_Q[0][i] << " ";
    }
    cout << endl;
    cout << "First 10 elements of real_Q[0]: " << endl;
    for (size_t i = 0; i < 10; i++) {
        cout << real_Q[0][i] << " ";
    }
    cout << endl;
    cout << "Rotary position embedding test ended." << endl;
}