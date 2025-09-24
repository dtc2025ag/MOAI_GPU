#include "include.cuh"

using namespace std;
using namespace std::chrono;
using namespace phantom::util;
using namespace phantom::arith;
using namespace moai;
// q(x) = Σ a[i] * x^i
static double eval_with_powers(const std::vector<double>& a, double x) {
    const int n = static_cast<int>(a.size()) - 1;
    double pow_x = 1.0;   // x^0
    double acc   = 0.0;
    for (int i = 0; i <= n; ++i) {
        acc += a[(size_t)i] * pow_x;
        pow_x *= x;       
    }
    return acc;
}

vector<double> silu_get_coefficients(int degree, double out_s){
    SiLU silu(degree);
    silu.set_input_range(-1.0, 1.0);
    silu.set_output_scale(out_s);
    silu.fit();
    auto power_coeffs = cheb_to_power_in_x(silu.get_coefficients(), silu.get_prescale(), silu.get_constant(), out_s);
    return power_coeffs;
}

void silu_coeff_verify()
{
    SiLU silu(63);
    double out_s = 1.0; // output scale
    silu.set_input_range(-1.0, 1.0);
    silu.set_output_scale(out_s);

    silu.fit();

    auto cheb = silu.get_coefficients();
    cout << "{\n";
    cout << "  \"cheb_coeffs\": [";
    for (size_t i=0; i<cheb.size();++i){
        cout << cheb[i] << (i+1<cheb.size() ? "," : "");
    }
    cout << "],\n";
    cout << "  \"prescale\": " << silu.get_prescale() << ",\n";
    cout << "  \"constant\": " << silu.get_constant() << "\n";
    cout << "}\n";

    auto power_coeffs = cheb_to_power_in_x(cheb, silu.get_prescale(), silu.get_constant(), out_s);
    for (size_t i=0;i<power_coeffs.size();++i){
        printf("a[%zu] = %.17g\n", i, power_coeffs[i]);
    }

    // ---- verify silu approximation (plaintext) ----


    auto horner = [&](double x)->double {
        // q(x) = sum_i a[i] * x^i
        double acc = 0.0;
        for (int i = (int)power_coeffs.size() - 1; i >= 0; --i) {
            acc = acc * x + power_coeffs[(size_t)i];
        }
        return acc;
    };

    auto silu_true = [](double x)->double {
        // 
        if (x > 20.0) return x;
        if (x < -20.0) return 0.0;
        double ex = std::exp(-x);
        return x / (1.0 + ex);
    };

    const double xmin = -1.0, xmax = 1.0;
    const int    N    = 10001;             
    const double step = (xmax - xmin) / (N - 1);

    double mse = 0.0, mae = 0.0;
    double max_abs_err = 0.0;
    double at_max_err_x = 0.0;
    double max_rel_err = 0.0;              
    const  double rel_floor = 1e-6;

    for (int i = 0; i < N; ++i) {
        double x = xmin + i * step;
        double qt = horner(x);
        double gt = silu_true(x);
        double err = qt - gt;
        double aerr = std::abs(err);
        mse += err * err;
        mae += aerr;
        if (aerr > max_abs_err) {
            max_abs_err = aerr;
            at_max_err_x = x;
        }
        if (std::abs(gt) > rel_floor) {
            double rel = aerr / std::abs(gt);
            if (rel > max_rel_err) max_rel_err = rel;
        }
    }
    mse /= N;
    mae /= N;
    double rmse = std::sqrt(mse);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "\n==== SiLU polynomial (power basis) verification ====\n";
    std::cout << "Range: [" << xmin << ", " << xmax << "], degree: "
              << (int)(power_coeffs.size()-1) << ", samples: " << N << "\n";
    std::cout << "MAE  = " << mae  << "\n";
    std::cout << "RMSE = " << rmse << "\n";
    std::cout << "MAX  = " << max_abs_err << " at x = " << at_max_err_x << "\n";
    std::cout << "Max relative err (|gt|>" << rel_floor << "): " << max_rel_err << "\n\n";

    auto show = [&](double x){
        double q = horner(x);
        double g = silu_true(x);
        double k = eval_with_powers(power_coeffs, x);
        std::cout << "x=" << std::setw(6) << x
                  << "  q(x)=" << std::setw(12) << q
                  << "  k(x)=" << std::setw(12) << k
                  << "  silu(x)=" << std::setw(12) << g
                  << "  err=" << std::setw(12) << (q-g) << "\n";
    };



    // show(-6.0);
    // show(-3.0);
    show(-1.0);
    show(-0.5);
    show( 0.0);
    show( 0.5);
    show( 1.0);
    // show( 3.0);
    // show( 6.0);
}


PhantomCiphertext silu(PhantomCiphertext &x,
                          PhantomContext &context, PhantomRelinKey &relin_keys, PhantomSecretKey &sk,
                          phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{

    PhantomCKKSEncoder phantom_encoder(context);
    // pack Phantom to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    // for test
    Decryptor decryptor(&context, &sk);

    double scale = x.scale();
    size_t slot_count = encoder.slot_count();

    // refer to orion
    vector<double> coeff_low_to_high = silu_get_coefficients(63, 1.0);


    vector<PhantomCiphertext> x_n(64);
    x_n[1] = x;
    double s0 = 1;
    PhantomPlaintext inv_e;
    encoder.encode(s0, x_n[1].params_id(), x_n[1].scale(), inv_e);
    evaluator.multiply_plain_inplace(x_n[1], inv_e);
    evaluator.rescale_to_next_inplace(x_n[1]);

    double inv_s0 = 1 / s0;
    double temps0 = inv_s0;
    for (int i = 1; i < 63; ++i)
    {
        coeff_low_to_high[i] *= temps0;
        temps0 *= inv_s0;
        // cout <<i<<" "<<coeff_high_to_low[i] <<" "<<temps0<<endl;
    }

    // compute x^2,4,8,16
    evaluator.square(x_n[1], x_n[2]);
    evaluator.relinearize_inplace(x_n[2], relin_keys);
    evaluator.rescale_to_next_inplace(x_n[2]);

    evaluator.square(x_n[2], x_n[4]);
    evaluator.relinearize_inplace(x_n[4], relin_keys);
    evaluator.rescale_to_next_inplace(x_n[4]);

    evaluator.square(x_n[4], x_n[8]);
    evaluator.relinearize_inplace(x_n[8], relin_keys);
    evaluator.rescale_to_next_inplace(x_n[8]);

    evaluator.square(x_n[8], x_n[16]);
    evaluator.relinearize_inplace(x_n[16], relin_keys);
    evaluator.rescale_to_next_inplace(x_n[16]);

    evaluator.square(x_n[16], x_n[32]);
    evaluator.relinearize_inplace(x_n[32], relin_keys);
    evaluator.rescale_to_next_inplace(x_n[32]);

    // cout <<"square. "<<endl;

    // compute x^3,5,9,17,33
    for (int i = 2; i < 33; i *= 2)
    {
        // cout<<i+1<<" ";
        evaluator.mod_switch_to_inplace(x_n[1], x_n[i].params_id());
        evaluator.multiply(x_n[1], x_n[i], x_n[i + 1]);
        evaluator.relinearize_inplace(x_n[i + 1], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 1]);
    }
    // compute x^6,10,18,34
    for (int i = 4; i < 33; i *= 2)
    {
        // cout<<i+2<<" ";
        evaluator.mod_switch_to_inplace(x_n[2], x_n[i].params_id());
        evaluator.multiply(x_n[2], x_n[i], x_n[i + 2]);
        evaluator.relinearize_inplace(x_n[i + 2], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 2]);
    }

    // compute x^7,11,19,35
    for (int i = 4; i < 33; i *= 2)
    {
        // cout<<i+3<<" ";
        evaluator.mod_switch_to_inplace(x_n[3], x_n[i].params_id());
        evaluator.multiply(x_n[3], x_n[i], x_n[i + 3]);
        evaluator.relinearize_inplace(x_n[i + 3], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 3]);
    }

    // compute x^12,20,36
    for (int i = 8; i < 33; i *= 2)
    {

        evaluator.mod_switch_to_inplace(x_n[4], x_n[i].params_id());
        evaluator.multiply(x_n[4], x_n[i], x_n[i + 4]);
        evaluator.relinearize_inplace(x_n[i + 4], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 4]);
    }

    // compute x^13,21,37
    for (int i = 8; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[5], x_n[i].params_id());
        evaluator.multiply(x_n[5], x_n[i], x_n[i + 5]);
        evaluator.relinearize_inplace(x_n[i + 5], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 5]);
    }

    // compute x^14,22,38
    for (int i = 8; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[6], x_n[i].params_id());
        evaluator.multiply(x_n[6], x_n[i], x_n[i + 6]);
        evaluator.relinearize_inplace(x_n[i + 6], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 6]);
    }

    // compute x^15,23,39
    for (int i = 8; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[7], x_n[i].params_id());
        evaluator.multiply(x_n[7], x_n[i], x_n[i + 7]);
        evaluator.relinearize_inplace(x_n[i + 7], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 7]);
    }

    // compute x^24,40
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[8], x_n[i].params_id());
        evaluator.multiply(x_n[8], x_n[i], x_n[i + 8]);
        evaluator.relinearize_inplace(x_n[i + 8], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 8]);
    }

    // compute x^25,41
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[9], x_n[i].params_id());
        evaluator.multiply(x_n[9], x_n[i], x_n[i + 9]);
        evaluator.relinearize_inplace(x_n[i + 9], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 9]);
    }

    // compute x^26,42
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[10], x_n[i].params_id());
        evaluator.multiply(x_n[10], x_n[i], x_n[i + 10]);
        evaluator.relinearize_inplace(x_n[i + 10], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 10]);
    }
    // compute x^27,43
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[11], x_n[i].params_id());
        evaluator.multiply(x_n[11], x_n[i], x_n[i + 11]);
        evaluator.relinearize_inplace(x_n[i + 11], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 11]);
    }
    // compute x^28,44
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[12], x_n[i].params_id());
        evaluator.multiply(x_n[12], x_n[i], x_n[i + 12]);
        evaluator.relinearize_inplace(x_n[i + 12], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 12]);
    }
    // compute x^29,45
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[13], x_n[i].params_id());
        evaluator.multiply(x_n[13], x_n[i], x_n[i + 13]);
        evaluator.relinearize_inplace(x_n[i + 13], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 13]);
    }
    // compute x^30,46
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[14], x_n[i].params_id());
        evaluator.multiply(x_n[14], x_n[i], x_n[i + 14]);
        evaluator.relinearize_inplace(x_n[i + 14], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 14]);
    }
    // compute x^31,47
    for (int i = 16; i < 33; i *= 2)
    {
        evaluator.mod_switch_to_inplace(x_n[15], x_n[i].params_id());
        evaluator.multiply(x_n[15], x_n[i], x_n[i + 15]);
        evaluator.relinearize_inplace(x_n[i + 15], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i + 15]);
    }
    // compute x^48-63
    for (int i = 16; i < 32; ++i){
        evaluator.mod_switch_to_inplace(x_n[i], x_n[32].params_id());
        evaluator.multiply(x_n[i], x_n[32], x_n[32 + i]);
        evaluator.relinearize_inplace(x_n[32 + i], relin_keys);
        evaluator.rescale_to_next_inplace(x_n[32 + i]);
    }



    PhantomPlaintext plain_result;
    vector<double> result;

    // decryptor.decrypt(x_n[63],plain_result);
    // encoder.decode(plain_result,result);
    // cout <<"x^63: ";
    // for (int ind = 0 ; ind < 10 ; ++ind){
    //     cout <<result[ind]<<" ";
    // }
    // cout <<endl;

    //  cout <<"x^n. "<<endl;

    //  double pt0 = 8*s0;
    //  double pt = pt0;
    //  double ptsum = 0.0;

    // compute \sum a_(24-i)x^i
    PhantomCiphertext res;
    for (int i = 1; i < 64; ++i)
    {
        evaluator.mod_switch_to_inplace(x_n[i], x_n[63].params_id());
        
            decryptor.decrypt(x_n[i],plain_result);
            encoder.decode(plain_result,result);
            cout <<"Ciphertext: x^"<<i<<": ";
            for (int ind = 0 ; ind < 1 ; ++ind){
                cout <<result[ind]<<" ";
            }
            //cout <<endl;
            // cout <<", coeff: "<<coeff_low_to_high[i]<<endl;
            // printf("coeff[%zu] = %.17g\n", i, coeff_low_to_high[i]);
        
        PhantomPlaintext coeff;
        encoder.encode(coeff_low_to_high[i], x_n[i].params_id(), x_n[i].scale(), coeff);
        evaluator.multiply_plain_inplace(x_n[i], coeff);
        evaluator.rescale_to_next_inplace(x_n[i]);
        x_n[i].scale() = scale;
        // decryptor.decrypt(x_n[i],plain_result);
        // encoder.decode(plain_result,result);
        // cout <<"Ciphertext: x^"<<i<<": ";
        // for (int ind = 0 ; ind < 1 ; ++ind){
        //     cout <<result[ind]<<" ";
        // }
        //cout <<endl;

        if (i == 1)
        {
        res = x_n[i];
        }
        else
        {
        evaluator.add_inplace(res, x_n[i]);
        }
        /*
            //cout <<"a: "<<coeff_high_to_low[24-i]<<", ";
            encoder.decode(coeff,result);
            cout <<", dcd(ecd(a)): ";
            for (int ind = 0 ; ind < 1 ; ++ind){
                cout <<result[ind]<<" ";
            }
            //cout <<endl;

            decryptor.decrypt(x_n[i],plain_result);
            encoder.decode(plain_result,result);
            cout <<", a_"<<24-i<<" * x^"<<i<<": ";
            for (int ind = 0 ; ind < 1 ; ++ind){
                cout <<result[ind]<<" ";
            }
            //cout <<endl;

            decryptor.decrypt(res,plain_result);
            encoder.decode(plain_result,result);
            cout <<", sum: ";
            for (int ind = 0 ; ind < 1 ; ++ind){
                cout <<result[ind]<<" ";
            }
            cout <<endl;

            cout <<"Plaintext: x^"<<i<<": "<<pt;
            double temppt = pt*coeff_high_to_low[24-i];
            cout <<", a_"<<24-i<<" * x^"<<i<<": "<<temppt;
            ptsum += temppt;
            pt *= pt0;
            cout <<", sum: "<<ptsum<<endl;
        */
    }

    // cout <<"sum. "<<endl;
    // compute res += ecd(a[24])
    PhantomPlaintext coeff0;
    encoder.encode(coeff_low_to_high[0], res.params_id(), res.scale(), coeff0);
    evaluator.add_plain_inplace(res, coeff0);
    // ptsum += coeff_high_to_low[24];
    // cout <<"pt result = "<<ptsum<<endl;

    return res;
}
