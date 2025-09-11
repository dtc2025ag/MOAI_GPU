#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

const int num_X = 256;
const int num_row = 128;
const int num_col = 768;
const int num_inter = 3072;
const double sqrt_d = 8.0;
int num_input = 11;

vector<vector<vector<double>>> input_x(num_X, vector<vector<double>>(num_row, vector<double>(num_col, 0)));

// paras for attention block
int col_W = 64;
int num_head = 12;
vector<vector<vector<double>>> WQ(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));
vector<vector<vector<double>>> WK(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));
vector<vector<vector<double>>> WV(num_head, vector<vector<double>>(num_col, vector<double>(col_W, 0.0)));

vector<vector<double>> bQ(num_head, vector<double>(col_W, 0.0));
vector<vector<double>> bK(num_head, vector<double>(col_W, 0.0));
vector<vector<double>> bV(num_head, vector<double>(col_W, 0.0));

vector<vector<double>> selfoutput(num_col, vector<double>(num_col, 0.0));
vector<double> selfoutput_bias(num_col, 0.0);
vector<double> layernorm1_gamma(num_col, 0.0);
vector<double> layernorm1_beta(num_col, 0.0);

vector<vector<double>> inter_weight(num_col, vector<double>(num_inter, 0.0));
vector<double> inter_bias(num_inter, 0.0);
vector<vector<double>> final_weight(num_inter, vector<double>(num_col, 0.0));
vector<double> final_bias(num_col, 0.0);
vector<double> layernorm2_gamma(num_col, 0.0);
vector<double> layernorm2_beta(num_col, 0.0);

void read_input()
{
    ifstream fin;
    fin.open("att_block_weights/embedded_inputs.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file embedded_inputs.txt" << endl;
    }
    char a;
    // the test file has 11 input vectors, length of each vector = 768

    for (int i = 0; i < num_input; ++i)
    {
        for (int j = 0; j < num_col - 1; ++j)
        {
            fin >> input_x[0][i][j];
            fin >> a;
        }
        fin >> input_x[0][i][num_col - 1];
    }
    fin.close();
    // for test
    // cout <<input_x[0][10][0]<<" "<<input_x[0][10][num_col-1]<<endl;
}

void read_weights()
{
    ifstream fin;
    // read matrix Q, size of Q = 12*64*768
    fin.open("att_block_weights/query_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file query_weight.txt" << endl;
    }
    char a;
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            for (int j = 0; j < num_col - 1; ++j)
            {
                fin >> WQ[k][j][i];
                fin >> a;
            }
            fin >> WQ[k][num_col - 1][i];
        }
    }

    fin.close();
    // for test
    // cout <<"WQ last element: "<<WQ[num_head-1][num_col-1][col_W-1]<<endl;

    // Q = Q/sqrt(d')
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < num_col; ++i)
        {
            for (int j = 0; j < col_W; ++j)
            {
                WQ[k][i][j] = WQ[k][i][j] / sqrt_d;
            }
        }
    }

    // read matrix K
    fin.open("att_block_weights/key_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file key_weight.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            for (int j = 0; j < num_col - 1; ++j)
            {
                fin >> WK[k][j][i];
                fin >> a;
            }
            fin >> WK[k][num_col - 1][i];
        }
    }
    fin.close();
    // for test
    // cout <<"WK last element: "<<WK[num_head-1][num_col-1][col_W-1]<<endl;

    // read matrix V
    fin.open("att_block_weights/value_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file value_weight.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            for (int j = 0; j < num_col - 1; ++j)
            {
                fin >> WV[k][j][i];
                fin >> a;
            }
            fin >> WV[k][num_col - 1][i];
        }
    }
    fin.close();
    // for test
    // cout <<"WV last element: "<<WV[num_head-1][num_col-1][col_W-1]<<endl;

    // read self output weight
    fin.open("self_output_weights/self_output_dense_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_dense_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        for (int i = 0; i < num_col - 1; ++i)
        {
            fin >> selfoutput[i][k];
            fin >> a;
        }
        fin >> selfoutput[num_col - 1][k];
    }
    fin.close();
    // cout <<"selfoutput last element: "<<selfoutput[num_col-1][num_col-1]<<endl;

    // read layernorm1 weight
    fin.open("self_output_weights/self_output_LayerNorm_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_LayerNorm_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm1_gamma[k];
    }
    fin.close();
    // cout <<"LayerNorm1 last element: "<<layernorm1_gamma[num_col-1]<<endl;
}

void read_bias()
{
    ifstream fin;
    // read bias Q, size of Q = 64
    fin.open("att_block_weights/query_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file query_bias.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            fin >> bQ[k][i];
        }
    }
    fin.close();
    // for test
    // cout <<"Q bias last element: "<<bQ[num_head-1][col_W-1]<<endl;

    // bias Q = bias Q / sqrt_d'
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            bQ[k][i] = bQ[k][i] / sqrt_d;
        }
    }

    fin.open("att_block_weights/key_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file key_bias.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            fin >> bK[k][i];
        }
    }
    fin.close();
    // for test
    // cout <<"K bias last element: "<<bK[num_head-1][col_W-1]<<endl;

    fin.open("att_block_weights/value_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file value_bias.txt" << endl;
    }
    for (int k = 0; k < num_head; ++k)
    {
        for (int i = 0; i < col_W; ++i)
        {
            fin >> bV[k][i];
        }
    }
    fin.close();
    // for test
    // cout <<"v bias last element: "<<bV[num_head-1][col_W-1]<<endl;

    // read self output bias
    fin.open("self_output_weights/self_output_dense_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_dense_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> selfoutput_bias[k];
    }
    fin.close();
    // for test
    // cout <<"selfoutput bias last element: "<<selfoutput_bias[num_col-1]<<endl;

    // read layernorm1 weight
    fin.open("self_output_weights/self_output_LayerNorm_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file self_output_LayerNorm_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm1_beta[k];
    }
    fin.close();
    // cout <<"LayerNorm1 bias last element: "<<layernorm1_beta[num_col-1]<<endl;
}

void read_feed_forward_param()
{
    ifstream fin;
    char a;
    // read inter weight
    fin.open("feed_forward_weights/intermediate_dense_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file intermediate_dense_weight.txt" << endl;
    }
    for (int k = 0; k < num_inter; ++k)
    {
        for (int i = 0; i < num_col - 1; ++i)
        {
            fin >> inter_weight[i][k];
            fin >> a;
        }
        fin >> inter_weight[num_col - 1][k];
    }
    fin.close();
    // cout <<"inter_weight last element: "<<inter_weight[num_col-1][num_inter-1]<<endl;

    // read inter bias
    fin.open("feed_forward_weights/intermediate_dense_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file intermediate_dense_bias.txt" << endl;
    }
    for (int k = 0; k < num_inter; ++k)
    {
        fin >> inter_bias[k];
    }
    fin.close();
    // cout <<"inter_bias last element: "<<inter_bias[num_inter-1]<<endl;

    // read final weight
    fin.open("feed_forward_weights/final_output_dense_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_dense_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        for (int i = 0; i < num_inter - 1; ++i)
        {
            fin >> final_weight[i][k];
            fin >> a;
        }
        fin >> final_weight[num_inter - 1][k];
    }
    fin.close();
    // cout <<"final_weight last element: "<<final_weight[num_inter-1][num_col-1]<<endl;

    // read final bias
    fin.open("feed_forward_weights/final_output_dense_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_dense_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> final_bias[k];
    }
    fin.close();
    // cout <<"final_bias last element: "<<final_bias[num_col-1]<<endl;

    // read layernorm2 weight
    fin.open("feed_forward_weights/final_output_LayerNorm_weight.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_LayerNorm_weight.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm2_gamma[k];
    }
    fin.close();
    // cout <<"LayerNorm2 weights last element: "<<layernorm2_gamma[num_col-1]<<endl;

    // read layernorm2 bias
    fin.open("feed_forward_weights/final_output_LayerNorm_bias.txt");
    if (!fin.is_open())
    {
        cout << "Cannot open file final_output_LayerNorm_bias.txt" << endl;
    }
    for (int k = 0; k < num_col; ++k)
    {
        fin >> layernorm2_beta[k];
    }
    fin.close();
    // cout <<"LayerNorm2 bias last element: "<<layernorm2_beta[num_col-1]<<endl;
}

#include <set>
std::vector<int> normalize_steps(const std::vector<int> &steps, int N)
{
    std::set<int> uniq; // 自动排序 + 去重

    for (int k : steps)
    {
        // 先映射到 [0, N)
        int t = ((k % N) + N) % N;

        if (t == 0)
            continue; // 0 对应恒等，不需要
        if (t > N / 2)
            t -= N; // 映射到 (-N/2, N/2]

        if (t == -N / 2)
            t = N / 2; // 统一成正代表
        if (t < 0)
            t = -t; // 只保留正代表（±k 等价）

        uniq.insert(t);
    }

    // 转换为 vector
    return std::vector<int>(uniq.begin(), uniq.end());
}

void single_layer_test()
{
    cout << "Task: test one layer of BERT in CKKS scheme: " << endl;

    read_input();
    read_weights();
    read_bias();
    read_feed_forward_param();
    cout << "Read input, weights, bias from txt files. " << endl;

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
    int remaining_level_att = 15;
    int boot_level = 14; // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level_att = remaining_level_att + boot_level;

    int remaining_level = 20;
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
    parms.set_sparse_slots(sparse_slots);
    double scale = pow(2.0, logp);
    // parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40,40, 40,
    //     40,40,40,40,40,40,40,40,40, 40,40,40,40,40,40,40,40,60}));
    // double scale = pow(2.0,40);

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
    // GaloisKeys gal_keys_boot;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    // PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);
    PhantomGaloisKey gal_keys_boot;

    // end

    // Encryptor encryptor(context, public_key);
    // Decryptor decryptor(context, secret_key);
    // CKKSEncoder encoder(context);
    // Evaluator evaluator(context, encoder);
    // size_t slot_count = encoder.slot_count();
    // cout <<slot_count<<endl;
    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    // prepare for bootstrapping
    //  Bootstrapper bootstrapper(
    //    loge,
    //    logn,
    //    logN - 1,
    //    total_level,
    //    scale,
    //    boundary_K,
    //    deg,
    //    scale_factor,
    //    inverse_deg,
    //    context,
    //    keygen,
    //    encoder,
    //    encryptor,
    //    decryptor,
    //    evaluator,
    //    relin_keys,
    //    gal_keys_boot);

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &phantom_encoder, &relin_keys, &gal_keys_boot, scale);

    // add on Aug 18, save GPU memory
    // vector<int> gal_vector;
    // gal_vector.push_back(0);
    // for (int i = 0; i < sparse_slots/num_X; ++i)
    // {
    //     gal_vector.push_back((i * num_X));
    //     // cout << (i * num_X) << " ";
    // }

    // gal_vector.push_back(0); // NEXUS
    // for (int i = 0; i < logN - 1; i++)
    // {
    //     gal_vector.push_back((1 << i));
    // }

    // // keygen.create_galois_keys(gal_vector, gal_keys);
    // ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_vector, gal_keys);
    // gal_keys = secret_key.create_galois_keys(context);

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

    cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    // ct-ct rotate steps
    // vector<int> gal_vector;
    // gal_steps_vector.push_back(0);
    // for (int i = 0; i < sparse_slots/num_X; ++i)
    // {
    //     gal_steps_vector.push_back((i * num_X));
    //     // cout << (i * num_X) << " ";
    // }

    // keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);

    // ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    gal_keys_boot = secret_key.create_galois_keys(context);
    // gal_keys = secret_key.create_galois_keys(context);
    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

    cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();

    

    struct timeval tstart1, tend1;

    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);
    vector<int> input_len(num_X, 0);
    input_len[0] = 11;
    vector<int> b_vec = bias_vec(input_len, num_X, num_row);
    // cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<endl;
    // cout <<"Modulus chain index for x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    vector<vector<vector<double>>>().swap(input_x);

    vector<PhantomCiphertext> enc_ecd_x_copy(num_col);
    for (int i = 0; i < num_col; ++i){
        enc_ecd_x_copy[i] = enc_ecd_x[i];
    }

    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x_copy[i]);
        }
    }

    // mod switch to remaining level
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < boot_level + (remaining_level - remaining_level_att); ++j)
        {
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }

    // mod switch to next level
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i)
    {
        evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
    }

    cout << "Modulus chain index before attention block: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;

    vector<vector<PhantomCiphertext>> att_block(num_head);

    gettimeofday(&tstart1, NULL);

    for (int i = 0; i < 1; ++i)
    {
        att_block[i] = single_att_block(enc_ecd_x, WQ[i], WK[i], WV[i], bQ[i], bK[i], bV[i],
                                        b_vec, num_input, context, relin_keys, gal_keys_boot, bootstrapper, num_X, secret_key, 16, 10);
        /*
        cout <<"Decrypt + decode result of ";
        cout <<i+1<<"-th head: "<<endl;
        for (int j = 0; j < att_block[i].size(); ++j){
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

    gettimeofday(&tend1, NULL);
    double att_block_time = tend1.tv_sec - tstart1.tv_sec + (tend1.tv_usec - tstart1.tv_usec) / 1000000.0;
    cout << "Attention block time = " << att_block_time << endl;
    append_csv_row("../single_layer_results.csv", "Attention Block", att_block_time);
    // cout <<"Modulus chain index for the result: "<< context.get_context_data(att_block[2][0].params_id()).chain_depth()<<endl;
// }
/*
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int k = 0; k < num_head; ++k){
        //cout <<k+1<<"-th head: "<<endl;
        for (int i = 0; i < att_block[k].size(); ++i){
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
// delete enc_ecd_x
    vector<PhantomCiphertext>().swap(enc_ecd_x);

    gettimeofday(&tstart1,NULL);

    int output_size = att_block[0].size();

    vector<PhantomCiphertext> att_output(num_head*output_size);

    for (int i = 0; i < num_head; ++i){
        for (int j = 0 ; j < output_size ; ++j){
            // att_output[i*output_size+j] = att_block[i][j];
            att_output[i*output_size+j] = att_block[0][j];
        }
    }

    cout <<"Concatenation. size of output of attention block = "<<num_head<<" * "<<output_size<<" = "<<att_output.size()<<endl;

    vector<vector<PhantomCiphertext>>().swap(att_block);

    //att_output * selfoutput + selfoutput_bias
    gettimeofday(&tstart1,NULL);
    cudaDeviceSynchronize();
    vector<PhantomCiphertext> att_selfoutput = ct_pt_matrix_mul_wo_pre_large(att_output, selfoutput, num_col, num_col, num_col, context);
    cudaDeviceSynchronize();
    gettimeofday(&tend1,NULL);
    double selfoutput_time_temp = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"selfoutput time(matrix_mul) = "<<selfoutput_time_temp<<endl;
    int att_selfoutput_size = att_selfoutput.size();
    //cout <<"num of ct in att_selfoutput = "<<att_selfoutput_size<<endl;
    for (int i = 0; i < num_col; ++i){
        PhantomPlaintext ecd_self_bias;
        vector<double> self_bias_vec(slot_count,0);
        for (int j = 0; j < slot_count; ++j){
            if(b_vec[j] == 1){
                self_bias_vec[j] = selfoutput_bias[i];
            }
        }
        encoder.encode(self_bias_vec, att_selfoutput[i].params_id(), att_selfoutput[i].scale(), ecd_self_bias);
        evaluator.mod_switch_to_inplace(ecd_self_bias, att_selfoutput[i].params_id());
        att_selfoutput[i].scale() = scale;
        ecd_self_bias.scale() = scale;
        evaluator.add_plain_inplace(att_selfoutput[i],ecd_self_bias);
    }

    gettimeofday(&tend1,NULL);
    double selfoutput_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"selfoutput time = "<<selfoutput_time<<endl;
    append_csv_row("../single_layer_results.csv", "SelfOutput", selfoutput_time);
    cout <<"Modulus chain index for the result: "<< context.get_context_data(att_selfoutput[0].params_id()).chain_depth()<<endl;

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
        while(context.get_context_data(att_selfoutput[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(att_selfoutput[i]);
        }
    }

    vector<PhantomCiphertext> rtn(att_selfoutput_size);

    //cout<<"bootstrapping start. "<<endl;

    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn[i*6+j],att_selfoutput[i*6+j]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time<<endl;
    append_csv_row("../single_layer_results.csv", "1st Bootstrapping", boot_time);
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn[0].params_id()).chain_depth()<<endl;

    //for (int i = 0; i < rtn.size(); ++i){
    //    evaluator.mod_switch_to_next_inplace(rtn[i]);
    //}
    //cout <<"Modulus chain index before layernorm: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;

    vector<PhantomCiphertext>().swap(att_selfoutput);

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
    //LayerNorm
    //cout <<"LayerNorm start. "<<endl;
    gettimeofday(&tstart1,NULL);

    //rtn+enc_ecd_x_copy
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_x_copy[i], rtn[i].params_id());
        evaluator.add_inplace(rtn[i],enc_ecd_x_copy[i]);
    }

    vector<PhantomCiphertext> layernorm_selfoutput = layernorm(rtn,layernorm1_gamma,layernorm1_beta, b_vec,
        context,relin_keys,secret_key);

    gettimeofday(&tend1,NULL);
    double layernorm_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"layernorm time = "<<layernorm_time<<endl;
    append_csv_row("../single_layer_results.csv", "LayerNorm1", layernorm_time);
    cout <<"Modulus chain index after layernorm: "<< context.get_context_data(layernorm_selfoutput[0].params_id()).chain_depth()<<endl;
    vector<PhantomCiphertext>().swap(enc_ecd_x_copy);
/*
    cout <<"Decrypt + decode result of layernorm: "<<endl;
    for (int i = 0; i < layernorm_selfoutput.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(layernorm_selfoutput[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(b_vec[ind] == 1){
                cout <<result[ind]<<", ";
            }
            else if(result[ind] >= 0.0001){
                cout <<"( "<<ind<<", "<<result[ind]<<"). ";
                continue;
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
        while(context.get_context_data(layernorm_selfoutput[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(layernorm_selfoutput[i]);
        }
    }

    vector<PhantomCiphertext> boot_layer(layernorm_selfoutput_size);

    //cout<<"bootstrapping start. "<<endl;

    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for
    rtn = vector<PhantomCiphertext>(layernorm_selfoutput_size);
    // vector<PhantomCiphertext>().swap(rtn);
    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn[i*6+j],layernorm_selfoutput[i*6+j]);
            boot_layer[i*6+j] = rtn[i*6+j];
        }
    }

    // #pragma omp parallel for

    for (int i = 0; i < rtn.size(); ++i) {
        for (int j = 0; j < 11; ++j){
            evaluator.mod_switch_to_next_inplace(rtn[i]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time2<<endl;
    append_csv_row("../single_layer_results.csv", "2nd Bootstrapping", boot_time2);
    //cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(boot_layer[0].params_id()).chain_depth()<<endl;

    vector<PhantomCiphertext>().swap(layernorm_selfoutput);

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

    cout <<"Modulus chain index before intermediate linear: "<< context.get_context_data(rtn[0].params_id()).chain_depth()<<endl;
    gettimeofday(&tstart1,NULL);

    vector<PhantomCiphertext> inter_output = ct_pt_matrix_mul_wo_pre_large(rtn, inter_weight, num_col, num_inter, num_col, context);
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
    // 线程数：不超过列数（避免空转）
    const int max_threads = omp_get_max_threads();
    const int nthreads = std::max(1, std::min(max_threads, 32));
    // std::cout << "nums of thread: " << nthreads << std::endl;

    // —— 准备每线程一个流（拥有型 wrapper） —— //
    if (nthreads == 1)
    {
        stream_pool.reserve(nthreads);
        stream_pool[0] = *phantom::util::global_variables::default_stream;
    }
    else if (stream_pool.size() < static_cast<size_t>(nthreads))
    {
        stream_pool.reserve(nthreads);
        for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
        {
        stream_pool.emplace_back(); // 默认构造：内部创建并持有一个新 CUDA 流
        }
    }
// —— 并行计算：每线程独立 Encoder/Evaluator（各自绑定线程私有的 PhantomCKKSEncoder） —— //
#pragma omp parallel num_threads(nthreads)
  {
    // cudaSetDevice(1);
    PhantomCKKSEncoder phantom_encoder_local(context);
    moai::Encoder encoder_local(&context, &phantom_encoder_local);
    moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

    const int tid = omp_get_thread_num();
    auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper

#pragma omp for schedule(static)    
        for (int i = 0; i < num_inter; ++i){
            PhantomPlaintext ecd_inter_bias;
            vector<double> inter_bias_vec(slot_count,0);
            for (int j = 0; j < slot_count; ++j){
                if(b_vec[j] == 1){
                    inter_bias_vec[j] = inter_bias[i];
                }
            }
            encoder_local.encode(inter_bias_vec, inter_output[i].params_id(), inter_output[i].scale(), ecd_inter_bias, stream);
            bridge_to_default(stream); // ★ 跨流桥接
            evaluator_local.mod_switch_to_inplace(ecd_inter_bias, inter_output[i].params_id());
            inter_output[i].scale() = scale;
            ecd_inter_bias.scale() = scale;
            evaluator_local.add_plain_inplace(inter_output[i],ecd_inter_bias);

        }
    cudaStreamSynchronize(stream.get_stream());
    }

    // for (int i = 0; i < num_inter; ++i){
    //     PhantomPlaintext ecd_inter_bias;
    //     vector<double> inter_bias_vec(slot_count,0);
    //     for (int j = 0; j < slot_count; ++j){
    //         if(b_vec[j] == 1){
    //             inter_bias_vec[j] = inter_bias[i];
    //         }
    //     }
    //     encoder.encode(inter_bias_vec, inter_output[i].params_id(), inter_output[i].scale(), ecd_inter_bias);
    //     evaluator.mod_switch_to_inplace(ecd_inter_bias, inter_output[i].params_id());
    //     inter_output[i].scale() = scale;
    //     ecd_inter_bias.scale() = scale;
    //     evaluator.add_plain_inplace(inter_output[i],ecd_inter_bias);

    // }

    gettimeofday(&tend1,NULL);
    double inter_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Inter layer time = "<<inter_time<<endl;
    append_csv_row("../single_layer_results.csv", "Intermediate Linear", inter_time);
    cout <<"Modulus chain index after inter layer: "<< context.get_context_data(inter_output[0].params_id()).chain_depth()<<endl;

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

    //2025.09.08, change to omp
    // const int max_threads = omp_get_max_threads();
    // const int nthreads = std::max(1, std::min(max_threads, 4));

    // —— 准备每线程一个流（拥有型 wrapper） —— //
    if (stream_pool.size() < static_cast<size_t>(nthreads))
    {
        stream_pool.reserve(nthreads);
        for (size_t i = stream_pool.size(); i < static_cast<size_t>(nthreads); ++i)
        {
            stream_pool.emplace_back(); // 默认构造：内部创建并持有一个新 CUDA 流
        }
    }
    if (nthreads == 1)
    {
        stream_pool[0] = *phantom::util::global_variables::default_stream;
    }

    vector<PhantomCiphertext> gelu_output(num_inter);

    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for
#pragma omp parallel num_threads(nthreads)
    {   
        // PhantomSecretKey secret_key_local(context);
        // PhantomRelinKey relin_keys_local = secret_key_local.gen_relinkey(context);

        const int tid = omp_get_thread_num();
        auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper
#pragma omp for schedule(static)
        for (int i = 0; i < 96; ++i)
        {
            for (int j = 0 ; j < 32; ++j)
            {
                gelu_output[i*32+j] = gelu_v2(inter_output[i*32+j],context,relin_keys,secret_key, stream);
            }
        }
        cudaStreamSynchronize(stream.get_stream());
    }

    gettimeofday(&tend1,NULL);
    double gelu_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"gelu time = "<<gelu_time<<endl;
    append_csv_row("../single_layer_results.csv", "GELU", gelu_time);
    cout <<"Modulus chain index for gelu: "<< context.get_context_data(gelu_output[0].params_id()).chain_depth()<<endl;

    vector<PhantomCiphertext>().swap(inter_output);
/*
    cout <<"Decrypt + decode result of intermediate_gelu: "<<endl;
    for (int i = 0; i < gelu_output.size(); ++i){
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

    vector<PhantomCiphertext> final_output = ct_pt_matrix_mul_wo_pre_w_mask(gelu_output, final_weight,b_vec, num_inter, num_col, num_inter, context);
    int final_output_size = final_output.size();
    //cout <<"num of ct in final_output = "<<final_output_size<<endl;

#pragma omp parallel num_threads(nthreads)
    {   
        PhantomCKKSEncoder phantom_encoder_local(context);
        moai::Encoder encoder_local(&context, &phantom_encoder_local);
        moai::Evaluator evaluator_local(&context, &phantom_encoder_local);

        const int tid = omp_get_thread_num();
        auto &stream = stream_pool[tid]; // ★ 引用，不要拷贝 wrapper
#pragma omp for schedule(static)
        for (int i = 0; i < num_col; ++i){
            PhantomPlaintext ecd_final_bias;
            vector<double> final_bias_vec(slot_count,0);
            for (int j = 0; j < slot_count; ++j){
                if(b_vec[j] == 1){
                    final_bias_vec[j] = final_bias[i];
                }
            }
            encoder_local.encode(final_bias_vec, final_output[i].params_id(), final_output[i].scale(), ecd_final_bias, stream);
            bridge_to_default(stream); // ★ 跨流桥接
            evaluator_local.mod_switch_to_inplace(ecd_final_bias, final_output[i].params_id());
            final_output[i].scale() = scale;
            ecd_final_bias.scale() = scale;
            evaluator_local.add_plain_inplace(final_output[i],ecd_final_bias);

        }
    cudaStreamSynchronize(stream.get_stream());
    }

    // for (int i = 0; i < num_col; ++i){
    //     PhantomPlaintext ecd_final_bias;
    //     vector<double> final_bias_vec(slot_count,0);
    //     for (int j = 0; j < slot_count; ++j){
    //         if(b_vec[j] == 1){
    //             final_bias_vec[j] = final_bias[i];
    //         }
    //     }
    //     encoder.encode(final_bias_vec, final_output[i].params_id(), final_output[i].scale(), ecd_final_bias);
    //     evaluator.mod_switch_to_inplace(ecd_final_bias, final_output[i].params_id());
    //     final_output[i].scale() = scale;
    //     ecd_final_bias.scale() = scale;
    //     evaluator.add_plain_inplace(final_output[i],ecd_final_bias);

    // }

    gettimeofday(&tend1,NULL);
    double final_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Final layer time = "<<final_time<<endl;
    append_csv_row("../single_layer_results.csv", "Final Linear", final_time);
    cout <<"Modulus chain index after final layer: "<< context.get_context_data(final_output[0].params_id()).chain_depth()<<endl;
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
        while(context.get_context_data(final_output[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(final_output[i]);
        }
    }

    //cout<<"bootstrapping start. "<<endl;
    vector<PhantomCiphertext> rtn2(768);
    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn2[i*6+j],final_output[i*6+j]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time3 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time3<<endl;
    append_csv_row("../single_layer_results.csv", "3rd Bootstrapping", boot_time3);
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn2[0].params_id()).chain_depth()<<endl;

   // for (int i = 0; i < rtn2.size(); ++i){
    //    evaluator.mod_switch_to_next_inplace(rtn2[i]);
   // }
    //cout <<"Modulus chain index before layernorm: "<< context.get_context_data(rtn2[0].parms_id())->chain_index()<<endl;

    vector<PhantomCiphertext>().swap(final_output);

    cout <<"LayerNorm start. "<<endl;
    gettimeofday(&tstart1,NULL);

    //rtn+enc_ecd_x_copy
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(boot_layer[i], rtn2[i].params_id());
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
    vector<PhantomCiphertext> layernorm_finaloutput = layernorm2(rtn2,layernorm2_gamma,layernorm2_beta,b_vec,
        context,relin_keys,secret_key);

    gettimeofday(&tend1,NULL);
    double layernorm_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"layernorm time = "<<layernorm_time2<<endl;
    append_csv_row("../single_layer_results.csv", "LayerNorm2", layernorm_time2);
    cout <<"Modulus chain index after layernorm: "<< context.get_context_data(layernorm_finaloutput[0].params_id()).chain_depth()<<endl;
    vector<PhantomCiphertext>().swap(rtn);

    // cout <<"Decrypt + decode result of one layer: "<<endl;
    // for (int i = 0; i < layernorm_finaloutput.size(); ++i){
    //     PhantomPlaintext plain_result;
    //     decryptor.decrypt(layernorm_finaloutput[i], plain_result);
    //     vector<double> result;
    //     encoder.decode(plain_result, result);
    //     cout <<i+1<<"-th ciphertext: ";
    //     for (int ind = 0 ; ind < slot_count ; ++ind){
    //         if(b_vec[ind] == 1){
    //             cout <<result[ind]<<", ";
    //         }
    //     }
    //     cout <<endl;
    // }

    // cout <<endl;

    //bootstrapping
    int layernorm2_size = layernorm_finaloutput.size();
    //mod switch the ciphertext to the lowest layer
    for (int i = 0; i < layernorm2_size; ++i){
        while(context.get_context_data(layernorm_finaloutput[i].params_id()).chain_depth() != 0){
        evaluator.mod_switch_to_next_inplace(layernorm_finaloutput[i]);
        }
    }

    //cout<<"bootstrapping start. "<<endl;
    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for
    rtn2 = vector<PhantomCiphertext>(layernorm2_size);
    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn2[i*6+j],layernorm_finaloutput[i*6+j]);
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time4 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time4<<endl;
    append_csv_row("../single_layer_results.csv", "4th Bootstrapping", boot_time4);
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn2[0].params_id()).chain_depth()<<endl;

    vector<PhantomCiphertext>().swap(layernorm_finaloutput);
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
    append_csv_row("../single_layer_results.csv", "Total time for one layer", total_time);
    append_csv_row("../single_layer_results.csv", "Amortized time", total_time/256.0);

}
