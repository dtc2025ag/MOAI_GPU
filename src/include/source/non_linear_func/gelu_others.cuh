using namespace std;
using namespace seal; 

Ciphertext gelu_v2(const Ciphertext & x, 
  const SEALContext& seal_context, const RelinKeys &relin_keys, const SecretKey& sk){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  //for test
  Decryptor decryptor(seal_context, sk);

  double scale = x.scale();
  size_t slot_count = encoder.slot_count();

  double coeff_high_to_low[] = {3.18006986e-24,5.70792114e-22,3.97205561e-20,1.31854608e-18,
  1.64153184e-17, -2.33052347e-16, -9.78309547e-15, -6.72238500e-14,
  1.43093357e-12, 2.41129634e-11, -4.00991558e-11, -3.06661368e-09,
 -1.00479838e-08, 2.05368974e-07, 1.25666834e-06, -7.76703686e-06,
 -6.75419265e-05, 1.62401656e-04, 1.97100905e-03, -1.70511673e-03,
 -3.22621248e-02, 7.22135066e-03, 3.39374355e-01, 4.92938360e-01, 1.21149468e-02};

  vector<Ciphertext> x_n(25);
  x_n[1] = x;
  double s0 = 0.1;
  Plaintext inv_e;
  encoder.encode(s0,x_n[1].parms_id(),x_n[1].scale(),inv_e);
  evaluator.multiply_plain_inplace(x_n[1],inv_e);
  evaluator.rescale_to_next_inplace(x_n[1]);

  double inv_s0 = 1/s0;
  double temps0 = inv_s0;
  for (int i = 23; i >=0 ; --i){
    coeff_high_to_low[i] *= temps0;
    temps0 *= inv_s0;
    //cout <<i<<" "<<coeff_high_to_low[i] <<" "<<temps0<<endl;
  }


  //compute x^2,4,8,16
  evaluator.square(x_n[1],x_n[2]);
  evaluator.relinearize_inplace(x_n[2],relin_keys);
  evaluator.rescale_to_next_inplace(x_n[2]);

  evaluator.square(x_n[2],x_n[4]);
  evaluator.relinearize_inplace(x_n[4],relin_keys);
  evaluator.rescale_to_next_inplace(x_n[4]);

  evaluator.square(x_n[4],x_n[8]);
  evaluator.relinearize_inplace(x_n[8],relin_keys);
  evaluator.rescale_to_next_inplace(x_n[8]);

  evaluator.square(x_n[8],x_n[16]);
  evaluator.relinearize_inplace(x_n[16],relin_keys);
  evaluator.rescale_to_next_inplace(x_n[16]);

 // cout <<"square. "<<endl;

  //compute x^3,5,9,17
  for (int i = 2; i < 17; i *= 2){
    //cout<<i+1<<" ";
    evaluator.mod_switch_to_inplace(x_n[1],x_n[i].parms_id());
    evaluator.multiply(x_n[1],x_n[i],x_n[i+1]);
    evaluator.relinearize_inplace(x_n[i+1],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[i+1]);
  }

  //compute x^6,10,18
  for (int i = 4; i < 17; i *= 2){
    //cout<<i+2<<" ";
    evaluator.mod_switch_to_inplace(x_n[2],x_n[i].parms_id());
    evaluator.multiply(x_n[2],x_n[i],x_n[i+2]);
    evaluator.relinearize_inplace(x_n[i+2],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[i+2]);
  }

  //compute x^7,11,19
  for (int i = 4; i < 17; i *= 2){
    //cout<<i+3<<" ";
    evaluator.mod_switch_to_inplace(x_n[3],x_n[i].parms_id());
    evaluator.multiply(x_n[3],x_n[i],x_n[i+3]);
    evaluator.relinearize_inplace(x_n[i+3],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[i+3]);
  }

  //compute x^12,20
  for (int i = 8; i < 17; i *= 2){

    evaluator.mod_switch_to_inplace(x_n[4],x_n[i].parms_id());
    evaluator.multiply(x_n[4],x_n[i],x_n[i+4]);
    evaluator.relinearize_inplace(x_n[i+4],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[i+4]);
  }

  //compute x^13,21
  for (int i = 8; i < 17; i *= 2){
    evaluator.mod_switch_to_inplace(x_n[5],x_n[i].parms_id());
    evaluator.multiply(x_n[5],x_n[i],x_n[i+5]);
    evaluator.relinearize_inplace(x_n[i+5],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[i+5]);
  }
  
  //compute x^14,22
  for (int i = 8; i < 17; i *= 2){
    evaluator.mod_switch_to_inplace(x_n[6],x_n[i].parms_id());
    evaluator.multiply(x_n[6],x_n[i],x_n[i+6]);
    evaluator.relinearize_inplace(x_n[i+6],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[i+6]);
  }

  //compute x^15,23
  for (int i = 8; i < 17; i *= 2){
    evaluator.mod_switch_to_inplace(x_n[7],x_n[i].parms_id());
    evaluator.multiply(x_n[7],x_n[i],x_n[i+7]);
    evaluator.relinearize_inplace(x_n[i+7],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[i+7]);
  }

  //compute x^24
  evaluator.mod_switch_to_inplace(x_n[8],x_n[16].parms_id());
  evaluator.multiply(x_n[8],x_n[16],x_n[24]);
  evaluator.relinearize_inplace(x_n[24],relin_keys);
  evaluator.rescale_to_next_inplace(x_n[24]);

  Plaintext plain_result;
  vector<double> result;

  //compute \sum a_(24-i)x^i
  Ciphertext res;
  for (int i = 1; i < 25; ++i){
    evaluator.mod_switch_to_inplace(x_n[i],x_n[24].parms_id());

    Plaintext coeff;
    encoder.encode(coeff_high_to_low[24-i],x_n[i].parms_id(),x_n[i].scale(),coeff);
    evaluator.multiply_plain_inplace(x_n[i],coeff);
    evaluator.rescale_to_next_inplace(x_n[i]);
    x_n[i].scale() = scale;
    if (i == 1){
      res = x_n[i];
    }
    else{
      evaluator.add_inplace(res,x_n[i]);
    }   
    
  }

  //cout <<"sum. "<<endl;
  //compute res += ecd(a[24])
  Plaintext coeff0;
  encoder.encode(coeff_high_to_low[24],res.parms_id(),res.scale(),coeff0);
  evaluator.add_plain_inplace(res,coeff0);


  return res;

}
