#include "include.cuh"

using namespace std;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;

std::pair<std::vector<PhantomCiphertext>, std::vector<PhantomCiphertext>> 
apply_rotary_pos_emb(PhantomContext &context, vector<PhantomCiphertext> &enc_Q, vector<PhantomCiphertext> &enc_K,const vector<vector<double>> cos, const vector<vector<double>> sin)
{
    // refer to transformers library
    // cos = cos.unsqueeze(unsqueeze_dim)
    // sin = sin.unsqueeze(unsqueeze_dim)
    // q_embed = (q * cos) + (rotate_half(q) * sin) --> q_embed = (q * cos) + q[64:] * (-sin) + q[:64] * sin
    // k_embed = (k * cos) + (rotate_half(k) * sin)

    // input: enc_Q, enc_K: 128 ciphertexts, cos, sin: vector(128, 64)
    
    // Q, K -> 128 ciphertexts, 128 rows and 256 batches
    // cos, sin -> 128 plaintexts, 128 rows and 256 batches [64, 32768]
    // Equals to {first 128 ciphertexts [multiply_plain] with cos element-wise} [add] {last 64 ciphertexts [multiply_plain] -sin, first 64 ciphertexts [multiply_plain] sin}
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);

    int n = enc_Q.size();
    vector<PhantomCiphertext> output_q(n);
    vector<PhantomCiphertext> output_k(n);

    vector<vector<double>> vec_cos(n, vector<double>(32768));
    vector<vector<double>> vec_sin(n, vector<double>(32768));
    vector<vector<double>> vec_neg_sin(n, vector<double>(32768));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            for (int k = 0; k < 256; k++)
            {
                vec_cos[i][k * 128 + j] = cos[j][i];
                vec_sin[i][k * 128 + j] = sin[j][i];
                vec_neg_sin[i][k * 128 + j] = -sin[j][i];
            }
        }

    }

    for (int i = 0; i < n; i++)
    {   
        PhantomCiphertext enc_cos_left_q;
        PhantomPlaintext plain_cos_i_q;
        encoder.encode(vec_cos[i], enc_Q[i].params_id(), enc_Q[i].scale(), plain_cos_i_q);
        evaluator.multiply_plain(enc_Q[i], plain_cos_i_q, enc_cos_left_q);

        PhantomCiphertext enc_cos_left_k;
        PhantomPlaintext plain_cos_i_k;
        encoder.encode(vec_cos[i], enc_K[i].params_id(), enc_K[i].scale(), plain_cos_i_k);
        evaluator.multiply_plain(enc_K[i], plain_cos_i_k, enc_cos_left_k);


        PhantomCiphertext enc_sin_right_q, enc_sin_right_k;
        PhantomPlaintext plain_sin_i_q, plain_sin_i_k;

        int r_i = (i + (n / 2)) % n;
        if (r_i >= (n / 2))
        {
            encoder.encode(vec_neg_sin[i], enc_Q[r_i].params_id(), enc_Q[i].scale(), plain_sin_i_q);
            evaluator.multiply_plain(enc_Q[r_i], plain_sin_i_q, enc_sin_right_q);

            encoder.encode(vec_neg_sin[i], enc_K[r_i].params_id(), enc_K[i].scale(), plain_sin_i_k);
            evaluator.multiply_plain(enc_K[r_i], plain_sin_i_k, enc_sin_right_k);
        }
        else
        {
            encoder.encode(vec_sin[i], enc_Q[r_i].params_id(), enc_Q[i].scale(), plain_sin_i_q);
            evaluator.multiply_plain(enc_Q[r_i], plain_sin_i_q, enc_sin_right_q);

            encoder.encode(vec_sin[i], enc_K[r_i].params_id(), enc_K[i].scale(), plain_sin_i_k);
            evaluator.multiply_plain(enc_K[r_i], plain_sin_i_k, enc_sin_right_k);
        }

        evaluator.add(enc_cos_left_q, enc_sin_right_q, output_q[i]);
        evaluator.rescale_to_next_inplace(output_q[i]);

        evaluator.add(enc_cos_left_k, enc_sin_right_k, output_k[i]);
        evaluator.rescale_to_next_inplace(output_k[i]);
    }
    
    return {std::move(output_q), std::move(output_k)};

}