using namespace std;
using namespace seal;

Ciphertext evalLine(Ciphertext x, Plaintext m, Plaintext c, const SEALContext& seal_context){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  double scale = x.scale();

  evaluator.mod_switch_to_inplace(m,x.parms_id());
  evaluator.multiply_plain_inplace(x,m);
  evaluator.rescale_to_next_inplace(x);
  evaluator.mod_switch_to_inplace(c,x.parms_id());
  x.scale() = scale;
  evaluator.add_plain_inplace(x,c);
  return x;
}

Ciphertext initGuess(Ciphertext x, const SEALContext& seal_context){
  CKKSEncoder encoder(seal_context);
  Plaintext a,b;
  encoder.encode(-1.29054537e-04, x.scale(), a);
  encoder.encode(1.29054537e-01, x.scale(), b);
  return evalLine(x,a,b,seal_context);
}

Ciphertext newtonIter(Ciphertext x, Ciphertext res, int iter, 
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  double scale = x.scale();

  for (int i = 0; i < iter; ++i){
    Plaintext three_half, neg_half;
    encoder.encode(1.5, scale, three_half);
    encoder.encode(-0.5, scale, neg_half);

    //x^2
    Ciphertext res_sq;
    evaluator.square(res, res_sq);
    evaluator.relinearize_inplace(res_sq,relin_keys);
    evaluator.rescale_to_next_inplace(res_sq);

    //-0.5*x*b
    Ciphertext res_x;
    evaluator.mod_switch_to_inplace(neg_half,x.parms_id());
    evaluator.multiply_plain(x,neg_half,res_x);
    evaluator.rescale_to_next_inplace(res_x);
    if(seal_context.get_context_data(res.parms_id())->chain_index()<
      seal_context.get_context_data(res_x.parms_id())->chain_index()){
      evaluator.mod_switch_to_inplace(res_x,res.parms_id());
    }
    else{
      evaluator.mod_switch_to_inplace(res,res_x.parms_id());
    }

    evaluator.multiply_inplace(res_x,res);
    evaluator.relinearize_inplace(res_x,relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    //-0.5*b*x^3
    evaluator.mod_switch_to_inplace(res_sq,res_x.parms_id());
    evaluator.multiply_inplace(res_x,res_sq);
    evaluator.relinearize_inplace(res_x,relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    //1.5*x
    evaluator.mod_switch_to_inplace(three_half, res.parms_id());
    evaluator.multiply_plain_inplace(res, three_half);
    evaluator.rescale_to_next_inplace(res);

    //-0.5*b*x^3 + 1.5*x
    evaluator.mod_switch_to_inplace(res, res_x.parms_id());
    res_x.scale()=scale;
    res.scale()=scale;
    evaluator.add_inplace(res, res_x);
  }
  return res;
}

Ciphertext goldSchmidtIter(Ciphertext v, Ciphertext y, int d, 
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  double scale = y.scale();
 // cout <<"scale = "<<log2(scale)<<endl;

  Plaintext constant;
  encoder.encode(0.5,scale,constant);

  //GoldSchmidt's algorithm
  evaluator.mod_switch_to_inplace(v,y.parms_id());
  Ciphertext x;
  evaluator.multiply(v,y,x);
  evaluator.relinearize_inplace(x,relin_keys);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(constant,y.parms_id());
  Ciphertext h;
  evaluator.multiply_plain(y,constant,h);
  evaluator.rescale_to_next_inplace(h);

  for (int i = 0; i < d; ++i){
    encoder.encode(0.5,scale,constant);
    Ciphertext r;
    evaluator.multiply(x,h,r);
    evaluator.relinearize_inplace(r,relin_keys);
    evaluator.rescale_to_next_inplace(r);
    r.scale() = scale;

    Ciphertext temp;
    evaluator.negate(r,temp);
    evaluator.mod_switch_to_inplace(constant,temp.parms_id());
    evaluator.add_plain(temp, constant, r);

    //x = x + x*r
    evaluator.mod_switch_to_inplace(x,r.parms_id());
    evaluator.multiply(x,r,temp);
    evaluator.relinearize_inplace(temp,relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    x.scale()=scale;
    temp.scale()=scale;
    evaluator.mod_switch_to_inplace(x,temp.parms_id());
    evaluator.add_inplace(x,temp);

    //h = h + h*r
    evaluator.mod_switch_to_inplace(h,r.parms_id());
    evaluator.multiply(h,r,temp);
    evaluator.relinearize_inplace(temp,relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    h.scale()=scale;
    temp.scale()=scale;
    evaluator.mod_switch_to_inplace(h,temp.parms_id());
    evaluator.add_inplace(h,temp);
  }
  encoder.encode(2.0,scale,constant);
  evaluator.mod_switch_to_inplace(constant, h.parms_id());
  evaluator.multiply_plain_inplace(h,constant);
  evaluator.rescale_to_next_inplace(h);

  return h;

}

Ciphertext invert_sqrt(Ciphertext x, int d_newt, int d_gold, 
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  

  Ciphertext res = initGuess(x, seal_context);
  Ciphertext y = newtonIter(x,res,d_newt,seal_context,relin_keys);
  Ciphertext sqrt_inv = goldSchmidtIter(x,y,d_gold,seal_context,relin_keys);
  return sqrt_inv;
}

vector<Ciphertext> layernorm(const vector<Ciphertext> & x, const vector<double>& gamma, const vector<double>& beta, const vector<int> & bias_vec,
  const SEALContext& seal_context, const RelinKeys &relin_keys, const SecretKey& sk){
  //algorithm may be different for different data range
  //depth need = 20 (current version)
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  //for test
  Decryptor decryptor(seal_context, sk);

  double scale = x[0].scale();
  size_t slot_count = encoder.slot_count();
  int num_ct = x.size();
  if(num_ct != 768){
    cout <<"ERROR: INPUT SIZE IS NOT CORRECT. "<<endl;
  }

  //compute u=(x0+x1+...+x768)
  Ciphertext ave_x=x[0];
  for (int i = 1; i < num_ct; ++i){
    evaluator.add_inplace(ave_x,x[i]);
  }

/*
  //for test
  Plaintext plain_result;
  vector<double> result;
  cout <<"  decrypt of sum_x: "<<endl;;
  decryptor.decrypt(ave_x,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<" ";
    }
  }
  cout <<endl;

*/
  //compute nx_0,...,nx_n
  Plaintext d;
  vector<double> ecd_n(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      ecd_n[i] = 768.0;
    }
  }
  encoder.encode(ecd_n,x[0].parms_id(),x[0].scale(),d);
  vector<Ciphertext> nx(num_ct);

  #pragma omp parallel for 

  for (int i = 0; i < num_ct; ++i){
    nx[i] = x[i];
    evaluator.multiply_plain_inplace(nx[i],d);
    evaluator.rescale_to_next_inplace(nx[i]);
    nx[i].scale()=scale;
  }

//  cout <<"Modulus chain index for nx1: "<< 
//    seal_context.get_context_data(nx[1].parms_id())->chain_index()<<endl;

/*
  cout <<"  decrypt of nx: "<<endl;
  for (int i = 0; i < num_ct; ++i){
    decryptor.decrypt(nx[i], plain_result);
    
    encoder.decode(plain_result, result);
    cout <<i<<"-th: ";
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<" ";
      }
    }
  cout <<endl;
  }
*/
  //compute var=((nx0-u)^2+...+(nx768-u)^2)/n^2
  //Ciphertext var = nx[0];
  evaluator.mod_switch_to_inplace(ave_x,nx[0].parms_id());
  ave_x.scale()=scale;

  //768 = 48*16, designed for multi-thread
  vector<Ciphertext> temp_var(48);
  
  #pragma omp parallel for 

  for (int i = 0; i < 48; ++i){
    Ciphertext temp_i;
    for(int j = 0 ; j < 16; ++j){
      //cout <<i<<" ";
      Ciphertext temp = nx[i*16+j];
      evaluator.sub_inplace(temp,ave_x);
      evaluator.square_inplace(temp);
    //evaluator.relinearize_inplace(temp,relin_keys);
    //evaluator.rescale_to_next_inplace(temp);
      if(j == 0){
        temp_i = temp;
      }
      else{
        evaluator.add_inplace(temp_i,temp);
      }
    }
    temp_var[i] = temp_i;
  }

  Ciphertext var = temp_var[0];
  for (int i = 1; i < 48; ++i){
    evaluator.add_inplace(var,temp_var[i]);
  }
  evaluator.relinearize_inplace(var,relin_keys);
  evaluator.rescale_to_next_inplace(var);

  vector<double> ecd_inv_n2(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      ecd_inv_n2[i] = 1/(768.0*768.0);
    }
  }
  Plaintext inv_d;
  encoder.encode(ecd_inv_n2, var.parms_id(), var.scale(), inv_d);
  evaluator.multiply_plain_inplace(var,inv_d);
  evaluator.rescale_to_next_inplace(var);

  cout <<"Modulus chain index for var: "<< 
    seal_context.get_context_data(var.parms_id())->chain_index()<<endl;

  cout <<"  decrypt of var: "<<endl;
  Plaintext plain_result;
  vector<double> result;
  decryptor.decrypt(var,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<" ";
    }
  }
  cout <<endl;

  //compute 1/sqrt(var)
  Ciphertext inv_sqrt_var = invert_sqrt(var,4,2,seal_context,relin_keys);

  //for test
  cout <<"Modulus chain index for invert sqrt: "<< 
    seal_context.get_context_data(inv_sqrt_var.parms_id())->chain_index()<<endl;
  
  cout <<"  decrypt of 1/sqrt(var): "<<endl;;
  decryptor.decrypt(inv_sqrt_var,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<" ";
    }
  }
  cout <<endl;

  //compute Gamma/sqrt(n)*(nxi-u)*inv+beta
  vector<Ciphertext> output(num_ct);
  evaluator.mod_switch_to_inplace(ave_x,inv_sqrt_var.parms_id());

  #pragma omp parallel for 

  for (int i = 0; i < num_ct; ++i){
    //cout<<i<<" ";
    output[i] = nx[i];
    evaluator.mod_switch_to_inplace(output[i],inv_sqrt_var.parms_id());
    evaluator.sub_inplace(output[i],ave_x);
    evaluator.multiply_inplace(output[i],inv_sqrt_var);
    evaluator.relinearize_inplace(output[i],relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_gamma_n(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
    if(bias_vec[j] == 1){
      ecd_gamma_n[j] = gamma[i]/sqrt(768.0);
    }
  }
    Plaintext ecd_gamma;
    encoder.encode(ecd_gamma_n,output[i].parms_id(),output[i].scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(output[i],ecd_gamma);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_betai(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
    if(bias_vec[j] == 1){
      ecd_betai[j] = beta[i];
    }
  }
    Plaintext ecd_beta;
    encoder.encode(ecd_betai,output[i].parms_id(),output[i].scale(),ecd_beta);
    evaluator.add_plain_inplace(output[i],ecd_beta);

  }

  return output;

}

vector<Ciphertext> layernorm2(const vector<Ciphertext> & x, const vector<double>& gamma, const vector<double>& beta, const vector<int> & bias_vec,
  const SEALContext& seal_context, const RelinKeys &relin_keys, const SecretKey& sk){
  //algorithm may be different for different data range
  //depth need = 20 (current version)
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  //for test
  Decryptor decryptor(seal_context, sk);

  double scale = x[0].scale();
  size_t slot_count = encoder.slot_count();
  int num_ct = x.size();
  if(num_ct != 768){
    cout <<"ERROR: INPUT SIZE IS NOT CORRECT. "<<endl;
  }

  //compute u=(x0+x1+...+x768)
  Ciphertext ave_x=x[0];
  for (int i = 1; i < num_ct; ++i){
    evaluator.add_inplace(ave_x,x[i]);
  }

/*
  //for test
  Plaintext plain_result;
  vector<double> result;
  cout <<"  decrypt of sum_x: "<<endl;;
  decryptor.decrypt(ave_x,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<" ";
    }
  }
  cout <<endl;
*/

  //compute nx_0,...,nx_n
  Plaintext d;
  vector<double> ecd_n(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      ecd_n[i] = 768.0;
    }
  }
  encoder.encode(ecd_n,x[0].parms_id(),x[0].scale(),d);
  vector<Ciphertext> nx(num_ct);

  #pragma omp parallel for 

  for (int i = 0; i < num_ct; ++i){
    nx[i] = x[i];
    evaluator.multiply_plain_inplace(nx[i],d);
    evaluator.rescale_to_next_inplace(nx[i]);
    nx[i].scale()=scale;
  }

//  cout <<"Modulus chain index for nx1: "<< 
//    seal_context.get_context_data(nx[1].parms_id())->chain_index()<<endl;

/*
  cout <<"  decrypt of nx: "<<endl;
  for (int i = 0; i < num_ct; ++i){
    decryptor.decrypt(nx[i], plain_result);
    
    encoder.decode(plain_result, result);
    cout <<i<<"-th: ";
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
          cout <<result[ind]<<" ";
      }
    }
  cout <<endl;
  }
*/
  //compute var=((nx0-u)^2+...+(nx768-u)^2)/n^2
  //Ciphertext var = nx[0];
  evaluator.mod_switch_to_inplace(ave_x,nx[0].parms_id());
  ave_x.scale()=scale;

  //768 = 48*16, designed for multi-thread
  vector<Ciphertext> temp_var(48);
  
  #pragma omp parallel for 

  for (int i = 0; i < 48; ++i){
    Ciphertext temp_i;
    for(int j = 0 ; j < 16; ++j){
      //cout <<i<<" ";
      Ciphertext temp = nx[i*16+j];
      evaluator.sub_inplace(temp,ave_x);
      evaluator.square_inplace(temp);
    //evaluator.relinearize_inplace(temp,relin_keys);
    //evaluator.rescale_to_next_inplace(temp);
      if(j == 0){
        temp_i = temp;
      }
      else{
        evaluator.add_inplace(temp_i,temp);
      }
    }
    temp_var[i] = temp_i;
  }

  Ciphertext var = temp_var[0];
  for (int i = 1; i < 48; ++i){
    evaluator.add_inplace(var,temp_var[i]);
  }
  evaluator.relinearize_inplace(var,relin_keys);
  evaluator.rescale_to_next_inplace(var);

  vector<double> ecd_inv_n2(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      ecd_inv_n2[i] = 1/(768.0*768.0*768.0);
    }
  }
  Plaintext inv_d;
  encoder.encode(ecd_inv_n2, var.parms_id(), var.scale(), inv_d);
  evaluator.multiply_plain_inplace(var,inv_d);
  evaluator.rescale_to_next_inplace(var);

  cout <<"Modulus chain index for var: "<< 
    seal_context.get_context_data(var.parms_id())->chain_index()<<endl;

  cout <<"  decrypt of var: "<<endl;
  Plaintext plain_result;
  vector<double> result;
  decryptor.decrypt(var,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<" ";
    }
  }
  cout <<endl;

  //compute 1/sqrt(var)
  Ciphertext inv_sqrt_var = invert_sqrt(var,4,2,seal_context,relin_keys);

  //for test
  cout <<"Modulus chain index for invert sqrt: "<< 
    seal_context.get_context_data(inv_sqrt_var.parms_id())->chain_index()<<endl;
  
  cout <<"  decrypt of 1/sqrt(var): "<<endl;;
  decryptor.decrypt(inv_sqrt_var,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<" ";
    }
  }
  cout <<endl;

  //compute Gamma/sqrt(n)*(nxi-u)*inv+beta
  vector<Ciphertext> output(num_ct);
  evaluator.mod_switch_to_inplace(ave_x,inv_sqrt_var.parms_id());

  #pragma omp parallel for 

  for (int i = 0; i < num_ct; ++i){
    //cout<<i<<" ";
    output[i] = nx[i];
    evaluator.mod_switch_to_inplace(output[i],inv_sqrt_var.parms_id());
    evaluator.sub_inplace(output[i],ave_x);
    evaluator.multiply_inplace(output[i],inv_sqrt_var);
    evaluator.relinearize_inplace(output[i],relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_gamma_n(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
    if(bias_vec[j] == 1){
      ecd_gamma_n[j] = gamma[i]/768.0;
    }
  }
    Plaintext ecd_gamma;
    encoder.encode(ecd_gamma_n,output[i].parms_id(),output[i].scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(output[i],ecd_gamma);
    evaluator.rescale_to_next_inplace(output[i]);

    vector<double> ecd_betai(slot_count,0);
    for (int j = 0; j < slot_count; ++j){
    if(bias_vec[j] == 1){
      ecd_betai[j] = beta[i];
    }
  }
    Plaintext ecd_beta;
    encoder.encode(ecd_betai,output[i].parms_id(),output[i].scale(),ecd_beta);
    evaluator.add_plain_inplace(output[i],ecd_beta);

  }

  return output;

}