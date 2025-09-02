#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

/*
inline std::string uint64_to_hex_string(std::uint64_t value)
{
    return seal::util::uint_to_hex_string(&value, std::size_t(1));
}
*/
#pragma once
void phantom_ckks_test()
{
    cout << "Task: test CKKS scheme in phantom library: " << endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
    // int secret_key_hamming_weight = 128;
    // parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    long sparse_slots = 16384;
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    cout << "Set encryption parameters and print" << endl;
    print_parameters(context);

    // KeyGenerator keygen(context);
    // SecretKey secret_key = keygen.secret_key();
    // PublicKey public_key;
    // keygen.create_public_key(public_key);
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);
    // GaloisKeys gal_keys;
    // keygen.create_galois_keys(gal_keys);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);

    Encryptor encryptor(&context, &public_key);

    Evaluator evaluator(&context, &phantom_encoder);

    Decryptor decryptor(&context, &secret_key);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    // std::vector<std::vector<double>> x(4096, std::vector<double>(768, 0.0));
    // // nexus encoding check
    // vector<PhantomPlaintext> a_pts;
    // a_pts.reserve(768);
    // std::chrono::_V2::system_clock::time_point start_encode = high_resolution_clock::now();
    // // cudaDeviceSynchronize();
    // for (int i = 0; i < 768; i++) {
    //     PhantomPlaintext pt;
    //     encoder.encode(x[i], scale, pt);
    //     a_pts.push_back(pt);
    // }
    // std::chrono::_V2::system_clock::time_point end_encode = high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> encode_duration = end_encode - start_encode;
    // // cudaDeviceSynchronize();
    // cout << "[DEBUG] Encoding duration: " << encode_duration.count() << " ms" << endl;
    // return ;

    vector<double> input;
    input.reserve(slot_count);
    double curr_point = 0;
    double step_size = 1.0 / (static_cast<double>(slot_count) - 1);
    for (size_t i = 0; i < slot_count; i++)
    {
        input.push_back(curr_point);
        curr_point += step_size;
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    cout << "Evaluating polynomial PI*x^3 + 0.4x + 1 ..." << endl;

    PhantomPlaintext plain_coeff3, plain_coeff1, plain_coeff0;
    encoder.encode(3.14159265, scale, plain_coeff3);
    encoder.encode(0.4, scale, plain_coeff1);
    encoder.encode(1.0, scale, plain_coeff0);

    PhantomPlaintext x_plain;
    cout << "Encode input vectors." << endl;
    encoder.encode(input, scale, x_plain);
    PhantomCiphertext x1_encrypted;
    encryptor.encrypt(x_plain, x1_encrypted);

    PhantomCiphertext x3_encrypted;
    cout << "Compute x^2 and relinearize:" << endl;
    evaluator.square(x1_encrypted, x3_encrypted);
    evaluator.relinearize_inplace(x3_encrypted, relin_keys);
    cout << "    + Scale of x^2 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    cout << "Rescale x^2." << endl;
    evaluator.rescale_to_next_inplace(x3_encrypted);
    cout << "    + Scale of x^2 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    cout << "Compute and rescale PI*x." << endl;
    PhantomCiphertext x1_encrypted_coeff3;
    evaluator.multiply_plain(x1_encrypted, plain_coeff3, x1_encrypted_coeff3);
    cout << "    + Scale of PI*x before rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;
    evaluator.rescale_to_next_inplace(x1_encrypted_coeff3);
    cout << "    + Scale of PI*x after rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;

    cout << "Compute, relinearize, and rescale (PI*x)*x^2." << endl;
    evaluator.multiply_inplace(x3_encrypted, x1_encrypted_coeff3);
    evaluator.relinearize_inplace(x3_encrypted, relin_keys);
    cout << "    + Scale of PI*x^3 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;
    evaluator.rescale_to_next_inplace(x3_encrypted);
    cout << "    + Scale of PI*x^3 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;

    cout << "Compute and rescale 0.4*x." << endl;
    evaluator.multiply_plain_inplace(x1_encrypted, plain_coeff1);
    cout << "    + Scale of 0.4*x before rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;
    evaluator.rescale_to_next_inplace(x1_encrypted);
    cout << "    + Scale of 0.4*x after rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;

    // cout << endl;
    // cout << "Parameters used by all three terms are different." << endl;
    // cout << "    + Modulus chain index for x3_encrypted: "
    //      << context.get_context_data(x3_encrypted.parms_id())->chain_index() << endl;
    // cout << "    + Modulus chain index for x1_encrypted: "
    //      << context.get_context_data(x1_encrypted.parms_id())->chain_index() << endl;
    // cout << "    + Modulus chain index for plain_coeff0: "
    //      << context.get_context_data(plain_coeff0.parms_id())->chain_index() << endl;
    // cout << endl;

    cout << endl;
    cout << "Parameters used by all three terms are different." << endl;
    cout << "    + Modulus chain index for x3_encrypted: "
         << context.get_context_data(x3_encrypted.params_id()).chain_depth() << endl;
    cout << "    + Modulus chain index for x1_encrypted: "
         << context.get_context_data(x1_encrypted.params_id()).chain_depth() << endl;
    cout << "    + Modulus chain index for plain_coeff0: "
         << context.get_context_data(plain_coeff0.chain_index()).chain_depth() << endl;
    cout << endl;

    cout << "The exact scales of all three terms are different:" << endl;
    ios old_fmt(nullptr);
    old_fmt.copyfmt(cout);
    cout << fixed << setprecision(10);
    cout << "    + Exact scale in PI*x^3: " << x3_encrypted.scale() << endl;
    cout << "    + Exact scale in  0.4*x: " << x1_encrypted.scale() << endl;
    cout << "    + Exact scale in      1: " << plain_coeff0.scale() << endl;
    cout << endl;
    cout.copyfmt(old_fmt);

    cout << "Normalize scales to 2^40." << endl;
    x3_encrypted.scale() = pow(2.0, 40);
    x1_encrypted.scale() = pow(2.0, 40);

    // cout << "Normalize encryption parameters to the lowest level." << endl;
    // parms_id_type last_parms_id = x3_encrypted.parms_id();
    // evaluator.mod_switch_to_inplace(x1_encrypted, last_parms_id);
    // evaluator.mod_switch_to_inplace(plain_coeff0, last_parms_id);

    cout << "Normalize encryption parameters to the lowest level." << endl;
    size_t last_parms_id = x3_encrypted.chain_index();
    evaluator.mod_switch_to_inplace(x1_encrypted, last_parms_id);
    evaluator.mod_switch_to_inplace(plain_coeff0, last_parms_id);

    cout << "Compute PI*x^3 + 0.4*x + 1." << endl;
    PhantomCiphertext encrypted_result;
    evaluator.add(x3_encrypted, x1_encrypted, encrypted_result);
    evaluator.add_plain_inplace(encrypted_result, plain_coeff0);

    PhantomPlaintext plain_result;
    cout << "Decrypt and decode PI*x^3 + 0.4x + 1." << endl;
    cout << "    + Expected result:" << endl;
    vector<double> true_result;
    for (size_t i = 0; i < input.size(); i++)
    {
        double x = input[i];
        true_result.push_back((3.14159265 * x * x + 0.4) * x + 1);
    }
    print_vector(true_result, 3, 7);

    decryptor.decrypt(encrypted_result, plain_result);
    vector<double> result;
    encoder.decode(plain_result, result);
    cout << "    + Computed result ...... Correct." << endl;
    print_vector(result, 3, 7);
}
