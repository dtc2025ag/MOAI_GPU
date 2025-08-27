using namespace std;
using namespace seal; 

uint64_t get_modulus(const Ciphertext &x, int k, 
  const SEALContext& seal_context){
  const vector<Modulus> &modulus = seal_context.get_context_data(x.parms_id())->parms().coeff_modulus();
  int sz = modulus.size();
  return modulus[sz-k].value();
}

Ciphertext eval_odd_deg9_poly(const vector<double> & a, const Ciphertext & x, 
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  double D = x.scale();
  size_t slot_count = encoder.slot_count();

  uint64_t p = get_modulus(x,1,seal_context);
  uint64_t q = get_modulus(x,2,seal_context);
  uint64_t r = get_modulus(x,3,seal_context);
  uint64_t s = get_modulus(x,4,seal_context);
  uint64_t t = get_modulus(x,5,seal_context);

  p = q;
  q = r;
  r = s;
  s = t;

  vector<double> a_scales(10,0);
  a_scales[1] = q;
  a_scales[3] = (double)p / D * q / D * r;
  a_scales[5] = (double)p / D * p / D * q / D * q / D * r;
  a_scales[7] = (double)p / D * p / D * q / D * q / D * q / D * r / D * s;
  a_scales[9] = (double)p / D * p / D * p / D * q / D * q / D * q / D * r / D * r / D * s;

  Ciphertext x2, x3, x6;

  Ciphertext tempx = x;
  evaluator.square(tempx,x2);
  evaluator.relinearize_inplace(x2,relin_keys);
  evaluator.rescale_to_next_inplace(x2);

  evaluator.mod_switch_to_next_inplace(tempx);
  evaluator.multiply(x2,tempx,x3);
  evaluator.relinearize_inplace(x3, relin_keys);
  evaluator.rescale_to_next_inplace(x3);

  evaluator.square(x3,x6);
  evaluator.relinearize_inplace(x6,relin_keys);
  evaluator.rescale_to_next_inplace(x6);

  Plaintext a1, a3, a5, a7, a9;

  //T1
  Ciphertext T1;
  double a5_scale = D/x2.scale()*p/x3.scale()*q;
  encoder.encode(a[5], x2.parms_id(), a5_scale, a5);
  evaluator.multiply_plain(x2,a5,T1);
  evaluator.rescale_to_next_inplace(T1);

  encoder.encode(a[3], T1.parms_id(), T1.scale(), a3);

  evaluator.add_plain_inplace(T1, a3);

  evaluator.multiply_inplace(T1, x3);
  evaluator.relinearize_inplace(T1, relin_keys);
  evaluator.rescale_to_next_inplace(T1);

  //T2
  Ciphertext T2;
  Plaintext a9_switched;
  double a9_scale = D / x3.scale() *r / x6.scale() *q;
  encoder.encode(a[9], x3.parms_id(), a9_scale, a9);
  evaluator.multiply_plain(x3, a9, T2);
  evaluator.rescale_to_next_inplace(T2);

  Ciphertext a7x;
  double a7_scale = T2.scale() /x.scale() *p;
  encoder.encode(a[7], x.parms_id(), a7_scale, a7);
  evaluator.multiply_plain(x,a7,a7x);
  evaluator.rescale_to_next_inplace(a7x);
  evaluator.mod_switch_to_inplace(a7x, T2.parms_id());

  double mid_scale = (T2.scale() + a7x.scale()) / 2;
  T2.scale() = a7x.scale() = mid_scale;
  evaluator.add_inplace(T2, a7x);
  evaluator.multiply_inplace(T2, x6);
  evaluator.relinearize_inplace(T2, relin_keys);
  evaluator.rescale_to_next_inplace(T2);

  //T3
  Ciphertext T3;
  encoder.encode(a[1], x.parms_id(), p, a1);
  evaluator.multiply_plain(x, a1, T3);
  evaluator.rescale_to_next_inplace(T3);

  double mid3_scale = (T1.scale() + T2.scale() + T3.scale()) / 3;
  T1.scale() = T2.scale() = T3.scale() = mid3_scale;

  Ciphertext dest = T2;
  evaluator.mod_switch_to_inplace(T1, dest.parms_id());
  evaluator.add_inplace(dest, T1);
  evaluator.mod_switch_to_inplace(T3, dest.parms_id());
  evaluator.add_inplace(dest, T3);

  return dest;
}

Ciphertext sgn_eval(const Ciphertext & x, int d_g, int d_f, double sgn_factor,
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  vector<double> f4_coeffs = {0, 315, 0, -420, 0, 378, 0, -180, 0, 35};
  vector<double> g4_coeffs = {0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623};
  // should be divided by (1 << 7)
  int f4_scale = (1 << 7);
  // should be divided by (1 << 10)
  int g4_scale = (1 << 10);

  vector<double> f4_coeffs_last(10, 0.0);
  vector<double> g4_coeffs_last(10, 0.0);

  for (int i = 0; i <= 9; i++) {
    f4_coeffs[i] /= (double)f4_scale;
    f4_coeffs_last[i] = f4_coeffs[i] * sgn_factor;

    g4_coeffs[i] /= (double)g4_scale;
    g4_coeffs_last[i] = g4_coeffs[i] * sgn_factor;
  }

  Ciphertext dest = x;

  for (int i = 0; i < d_g; i++) {
    if (i == d_g - 1) {
      dest = eval_odd_deg9_poly(g4_coeffs_last, dest, seal_context, relin_keys);
    } else {
      dest = eval_odd_deg9_poly(g4_coeffs, dest, seal_context, relin_keys);
    }
  }
  for (int i = 0; i < d_f; i++) {
    if (i == d_f - 1) {
      dest = eval_odd_deg9_poly(f4_coeffs_last, dest, seal_context, relin_keys);
    } else {
      dest = eval_odd_deg9_poly(f4_coeffs, dest, seal_context, relin_keys);
    }
  }
  return dest;

}

Ciphertext gelu(const Ciphertext & x, 
  const SEALContext& seal_context, const RelinKeys &relin_keys, const SecretKey& sk){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  //for test
  Decryptor decryptor(seal_context, sk);

  double scale = x.scale();
  size_t slot_count = encoder.slot_count();

  //encode -3.5;
  vector<double> ss1(slot_count,-3.5);
  Plaintext p0;
  encoder.encode(ss1, x.parms_id(), x.scale(), p0);
  vector<double> ().swap(ss1);

  //encode 3.5
  vector<double> ss2(slot_count,3.5);
  Plaintext p1;
  encoder.encode(ss2,x.parms_id(),x.scale(),p1);
  vector<double> ().swap(ss2);

  //encode 1/8.5 //UPDATE Apr 17: 8.5->20.5
  vector<double> ss3(slot_count, 1.0/60);
  Plaintext delta;
  encoder.encode(ss3,x.parms_id(),x.scale(),delta);
  vector<double> ().swap(ss3);

  //b0 = delta*(x-p0)
  Ciphertext b0;
  evaluator.sub_plain(x,p0,b0);
  evaluator.multiply_plain_inplace(b0,delta);
  evaluator.rescale_to_next_inplace(b0);

  //b1 = delta*(x-p1)
  Ciphertext b1;
  evaluator.sub_plain(x, p1, b1);
  evaluator.multiply_plain_inplace(b1, delta);
  evaluator.rescale_to_next_inplace(b1);

/*
  //for test
  Plaintext plain_result;
  vector<double> result;
  decryptor.decrypt(b0,plain_result);
  encoder.decode(plain_result,result);
  //cout <<"b0: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
        cout <<result[ind]<<" ";
  }
  cout <<endl;

  decryptor.decrypt(b1,plain_result);
  encoder.decode(plain_result,result);
  //cout <<"b1: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
        cout <<result[ind]<<" ";
  }
  cout <<endl;
*/
  //cout <<"Modulus chain index before sgn_eval: "<< seal_context.get_context_data(b0.parms_id())->chain_index()<<endl;

  b0 = sgn_eval(b0, 2, 2, 0.5, seal_context, relin_keys);
  b1 = sgn_eval(b1, 2, 2, 0.5, seal_context, relin_keys);

  //cout <<"Modulus chain index after sgn_eval: "<< seal_context.get_context_data(b0.parms_id())->chain_index()<<endl;
/*
  decryptor.decrypt(b0,plain_result);
  encoder.decode(plain_result,result);
  cout <<"b0: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
        cout <<result[ind]<<" ";
  }
  cout <<endl;

  decryptor.decrypt(b1,plain_result);
  encoder.decode(plain_result,result);
  cout <<"b1: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
        cout <<result[ind]<<" ";
  }
  cout <<endl;
*/
  //cout <<"sign eval. "<<endl;

  vector<double> ss4(slot_count, 0.5);
  Plaintext zero_point_five;
  encoder.encode(ss4, b1.parms_id(), b1.scale(), zero_point_five);
  vector<double> ().swap(ss4);

  Ciphertext a1, a2;

  //a1 = b0-b1, a2 = b1+0.5
  evaluator.sub(b0, b1, a1);
  evaluator.add_plain(b1, zero_point_five, a2);

  //cout <<"Modulus chain index of a1,a2: "<< seal_context.get_context_data(a1.parms_id())->chain_index()<<endl;
  //cout <<"Modulus chain index before Ax: "<< seal_context.get_context_data(x.parms_id())->chain_index()<<endl;
  Ciphertext x_2;
  evaluator.square(x,x_2);
  evaluator.relinearize_inplace(x_2,relin_keys);
  evaluator.rescale_to_next_inplace(x_2);

  Ciphertext x_4;
  evaluator.square(x_2,x_4);
  evaluator.relinearize_inplace(x_4,relin_keys);
  evaluator.rescale_to_next_inplace(x_4);

  Ciphertext x_6;
  evaluator.mod_switch_to_inplace(x_2, x_4.parms_id());
  evaluator.multiply(x_2,x_4,x_6);
  evaluator.relinearize_inplace(x_6, relin_keys);
  evaluator.rescale_to_next_inplace(x_6);

  Ciphertext x_8;
  evaluator.square(x_4,x_8);
  evaluator.relinearize_inplace(x_8,relin_keys);
  evaluator.rescale_to_next_inplace(x_8);

  Ciphertext x_10;
  evaluator.mod_switch_to_inplace(x_2, x_8.parms_id());
  evaluator.multiply(x_2, x_8, x_10);
  evaluator.relinearize_inplace(x_10, relin_keys);
  evaluator.rescale_to_next_inplace(x_10);

  Ciphertext x_12;
  evaluator.square(x_6, x_12);
  evaluator.relinearize_inplace(x_12, relin_keys);
  evaluator.rescale_to_next_inplace(x_12);

  double A[] = {2.25775755e-04, 0.5, 3.96880960e-01, -6.37042698e-02, 8.38841647e-03, -7.17830961e-04, 3.49617829e-05, -7.26059653e-07};
  vector<Plaintext> coeff_A(8);
  for (size_t i = 0; i < coeff_A.size(); i++) {
    encoder.encode(A[i], scale, coeff_A[i]);
  }
  vector<Ciphertext> cts(8);
  cts[1] = x;
  cts[2] = x_2;
  cts[3] = x_4;
  cts[4] = x_6;
  cts[5] = x_8;
  cts[6] = x_10;
  cts[7] = x_12;

  //cout <<"x,...,x^12. "<<endl;
  // Ax = A[0]+A[1]x+A[2]x^2+A[3]x^4+A[4]x^6+A[5]x^8+A[6]x^10+A[7]x^12
  for (size_t i = 1; i < coeff_A.size(); i++) {
    evaluator.mod_switch_to_inplace(coeff_A[i], cts[i].parms_id());
    evaluator.multiply_plain_inplace(cts[i], coeff_A[i]);
    evaluator.rescale_to_next_inplace(cts[i]);
    cts[i].scale() = scale;
  }

  Ciphertext Ax = cts[cts.size() - 1];

  for (size_t i = 1; i < coeff_A.size() - 1; i++) {
    evaluator.mod_switch_to_inplace(cts[i], Ax.parms_id());
    evaluator.add_inplace(Ax, cts[i]);
  }

  evaluator.mod_switch_to_inplace(coeff_A[0], Ax.parms_id());
  evaluator.add_plain_inplace(Ax, coeff_A[0]);

  //cout <<"Modulus chain index after Ax: "<< seal_context.get_context_data(Ax.parms_id())->chain_index()<<endl;

  Ciphertext s1, s2;
  // // cout << Ax.scale() << " " << Bx.scale() << " " << a1.scale() << endl;
  evaluator.mod_switch_to_inplace(Ax, a1.parms_id());
  evaluator.multiply(Ax, a1, s1);
  evaluator.relinearize_inplace(s1, relin_keys);
  evaluator.rescale_to_next_inplace(s1);
  //s1 = s1*a1
  evaluator.mod_switch_to_inplace(a1,s1.parms_id());
  evaluator.multiply(s1, a1, s1);
  evaluator.relinearize_inplace(s1, relin_keys);
  evaluator.rescale_to_next_inplace(s1);
/*
  decryptor.decrypt(Ax,plain_result);
  encoder.decode(plain_result,result);
  cout <<"Ax: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
        cout <<result[ind]<<" ";
  }
  cout <<endl;

  decryptor.decrypt(a1,plain_result);
  encoder.decode(plain_result,result);
  cout <<"a1: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
        cout <<result[ind]<<" ";
  }
  cout <<endl;
*/


  Ciphertext tempx = x;
  evaluator.mod_switch_to_inplace(tempx, a2.parms_id());
  evaluator.multiply(tempx, a2, s2);
  evaluator.relinearize_inplace(s2, relin_keys);
  evaluator.rescale_to_next_inplace(s2);
/*
  decryptor.decrypt(a2,plain_result);
  encoder.decode(plain_result,result);
  cout <<"a2: ";
  for (int ind = 0 ; ind < 10 ; ++ind){
        cout <<result[ind]<<" ";
  }
  cout <<endl;
*/
  s1.scale() = scale;
  s2.scale() = scale;
  evaluator.mod_switch_to_inplace(s2, s1.parms_id());
  Ciphertext res;
  evaluator.add(s1, s2, res);

  return res;

}

vector<double> gelu_plain(vector<double> &input) {
  vector<double> output;
  output.reserve(input.size());

  for (double x : input) {
    double gelu_x = 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
    output.push_back(gelu_x);
  }

  return output;
}