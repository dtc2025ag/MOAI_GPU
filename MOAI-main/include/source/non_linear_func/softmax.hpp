#include <chrono>
using namespace chrono;
#include "Bootstrapper.h"
#include "ckks_evaluator.h"

using namespace std;
using namespace seal;

Ciphertext exp(const Ciphertext & x, const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  Plaintext inverse_128;
  encoder.encode(0.0078125,x.parms_id(),x.scale(),inverse_128);
  //evaluator.mod_switch_to_inplace(inverse_128,x.parms_id());
  //cout <<"encode 0.0078125"<<endl;

  Ciphertext output;
  evaluator.multiply_plain(x, inverse_128, output);
  evaluator.rescale_to_next_inplace(output);
  //cout <<"x*0.0078125"<<endl;
  //cout <<log2(output.scale())<<endl;

  Plaintext one;
  encoder.encode(1.0, output.parms_id(), output.scale(), one);
  //cout <<"encode 1"<<endl;
  //Ciphertext res;
  evaluator.add_plain_inplace(output, one);
  //cout <<"x*0.0078125+1"<<endl;
  //evaluator.rescale_to_next_inplace(output);
  //cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output.parms_id())->chain_index()<<endl;



  //compute output^128
  for (int i = 0; i < log2(128); ++i){
    //cout <<i<<endl;
    evaluator.square_inplace(output);
    evaluator.relinearize_inplace(output, relin_keys);
    evaluator.rescale_to_next_inplace(output);
  }
  //cout <<"(x*0.0078125+1)^128"<<endl;
  //cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output.parms_id())->chain_index()<<endl;

  return output;

}

Ciphertext inverse(const Ciphertext & x, const SEALContext& seal_context, 
  const RelinKeys &relin_keys, int iter) {
  //by default, iter = 4 (from Nexus)
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  Plaintext one;
  encoder.encode(1.0,x.parms_id(), x.scale(), one);

  Ciphertext y;
  evaluator.sub_plain(x, one, y);
  evaluator.negate_inplace(y);

  Ciphertext tmp;
  evaluator.add_plain(y, one, tmp);

  Ciphertext res = tmp;
  for (int i = 0; i < iter; ++i){
    evaluator.square_inplace(y);
    evaluator.relinearize_inplace(y,relin_keys);
    evaluator.rescale_to_next_inplace(y);

    //cout <<"y scale = "<<log2(y.scale())<<" , one scale = "<<log2(one.scale())<<endl;
    encoder.encode(1.0,y.parms_id(), y.scale(), one);
    evaluator.add_plain(y, one, tmp);

    evaluator.mod_switch_to_inplace(res, tmp.parms_id());
    evaluator.multiply_inplace(res,tmp);
    evaluator.relinearize_inplace(res, relin_keys);
    evaluator.rescale_to_next_inplace(res);
  }

  return res;
}



vector<Ciphertext> softmax(const vector<Ciphertext> & enc_X, const vector<int> & bias_vec, int input_num, const SEALContext& seal_context, 
  const RelinKeys &relin_keys, int iter, const SecretKey & sk){

  int num = enc_X.size();
  //cout <<"number of ct in output = "<<num<<endl;
  vector<Ciphertext> output(num);

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  //for test
  Decryptor decryptor(seal_context, sk);
  int slot_count = encoder.slot_count();
  //cout <<"slot count = "<<slot_count<<endl;
  int num_batch = slot_count/128;
  //cout <<"number of batch = "<<num_batch<<endl;

  //compute x_ij - 8
  vector<Ciphertext> enc_x_minus(num);

  double minus_index = 8.1;
  vector<double> minus(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      minus[i] = minus_index;
    }
  }

  for (int i = 0; i < num; ++i){
    enc_x_minus[i] = enc_X[i];
    //for slot with value neq 0, minus 8
    //case 0: first line
    if(i == 0){
      Plaintext one;
      encoder.encode(minus,enc_x_minus[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,enc_x_minus[i].parms_id());
      evaluator.sub_plain_inplace(enc_x_minus[i],one);
    }
    //case1: all zero row
    else if(i > input_num && i <= (num-input_num)){

    }
    //case2: 0 - input_num line
    else if(i <= input_num){
      vector<double>temps1(slot_count,0);
      int index = num_batch * (input_num-i);
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i < index){
          temps1[i] = minus_index;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 1;
      Plaintext one;
      encoder.encode(temps1,enc_x_minus[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,enc_x_minus[i].parms_id());
      evaluator.sub_plain_inplace(enc_x_minus[i],one);
    }
    //case3: num-input - num line
    else if(i > num-input_num){
      vector<double>temps1(slot_count,0);
      int index = (num-i) * num_batch;
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i >= index){
          //cout <<i<<endl;
          temps1[i] = minus_index;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 0;
      Plaintext one;
      encoder.encode(temps1,enc_x_minus[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,enc_x_minus[i].parms_id());
      evaluator.sub_plain_inplace(enc_x_minus[i],one);
    }
    //else{
    //  cout <<"ERROR in computing e^x. "<<endl;
   // }

  }



  //compute e^x_ij
  vector<Ciphertext> exp_x(num);
  
  vector<double> s1(slot_count,1);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      s1[i] = 0;
    }
  }

  #pragma omp parallel for 

  for (int i = 0; i < num; ++i){
    exp_x[i] = exp(enc_x_minus[i],seal_context,relin_keys);

    //for slot with value 0, minus 1
    //case 0: first line
    if(i == 0){
      Plaintext one;
      encoder.encode(s1,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.sub_plain_inplace(exp_x[i],one);
    }
    //case1: all zero row
    else if(i > input_num && i <= (num-input_num)){
      Plaintext one;
      encoder.encode(1,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.sub_plain_inplace(exp_x[i],one);
    }
    //case2: 0 - input_num line
    else if(i <= input_num){
      vector<double>temps1(slot_count,1);
      int index = num_batch * (input_num-i);
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i < index){
          temps1[i] = 0;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 1;
      Plaintext one;
      encoder.encode(temps1,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.sub_plain_inplace(exp_x[i],one);
    }
    //case3: num-input - num line
    else if(i > num-input_num){
      vector<double>temps1(slot_count,1);
      int index = (num-i) * num_batch;
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i >= index){
          //cout <<i<<endl;
          temps1[i] = 0;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 0;
      Plaintext one;
      encoder.encode(temps1,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.sub_plain_inplace(exp_x[i],one);
    }
    //else{
    //  cout <<"ERROR in computing e^x. "<<endl;
   // }

  }
  cout <<"    Modulus chain for e^x: "<< seal_context.get_context_data(exp_x[0].parms_id())->chain_index()<<endl;
  //cout <<log2(exp_x[0].scale())<<endl;
  Plaintext plain_result;
  vector<double> result;

/*
  cout <<"TEST result during softmax: "<<endl;
  
  cout <<"  decrypt of e^x: "<<endl;
  for (int i = 0; i < num; ++i){
    decryptor.decrypt(exp_x[i], plain_result);
    
    encoder.decode(plain_result, result);
    cout <<i<<"-th: ";
    for (int ind = 0 ; ind < slot_count ; ++ind){
      if(bias_vec[ind] == 1){
        if(result[ind] > 0.0000001){
          cout <<result[ind]<<" ";
        }
        else{
          cout <<"0 ";
        }
      }
    }
  cout <<endl;
  }
 */

  //compute /sum e^x_j
  Ciphertext sum_exp_x = exp_x[0];
  for (int i = 1; i < num; ++i){
    evaluator.add_inplace(sum_exp_x,exp_x[i]);
  }

  cout <<"  decrypt of sum_exp_(x-8): "<<endl;;
  decryptor.decrypt(sum_exp_x,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<result[ind]<<endl;
    }
  }
  cout <<endl;

  //compute Inv(sum_exp_x)
  Ciphertext inv_sum = inverse(sum_exp_x,seal_context,relin_keys,iter);
 // cout <<"Modulus chain for inv(sum): "<< seal_context.get_context_data(inv_sum.parms_id())->chain_index()<<endl;
  //cout <<log2(inv_sum.scale())<<endl;

  cout <<"  decrypt of inv(sum_exp_(x-8)): "<<endl;
  decryptor.decrypt(inv_sum,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
       cout <<result[ind]<<endl;
    }
  }
  cout <<endl;

  #pragma omp parallel for 

  for (int i = 0; i < num; ++i){
    evaluator.mod_switch_to_inplace(exp_x[i],inv_sum.parms_id());
    evaluator.multiply(exp_x[i],inv_sum,output[i]);
    evaluator.relinearize_inplace(output[i],relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);
  } 

  return output;

}

vector<Ciphertext> softmax_boot(const vector<Ciphertext> & enc_X, const vector<int> & bias_vec, int input_num, const SEALContext& seal_context, 
  const RelinKeys &relin_keys, int iter, const SecretKey & sk, Bootstrapper& bootstrapper_att, int layer_id){

  int num = enc_X.size();
  double scale = enc_X[0].scale();
  //cout <<"number of ct in output = "<<num<<endl;
  vector<Ciphertext> output(num);

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  //for test
  Decryptor decryptor(seal_context, sk);
  int slot_count = encoder.slot_count();
  //cout <<"slot count = "<<slot_count<<endl;
  int num_batch = slot_count/128;
  //cout <<"number of batch = "<<num_batch<<endl;
  vector<double> minus_index_vec = {7.5, 9.9, 13.6, 13.3, 9.5, 8, 10.3, 9, 9, 9, 11, 7};

  //compute x_ij - 8
  vector<Ciphertext> enc_x_minus(num);

  double minus_index = minus_index_vec[layer_id];
  cout <<"softmax max = "<<minus_index<<endl;
  vector<double> minus(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      minus[i] = minus_index;
    }
  }

  #pragma omp parallel for 

  for (int i = 0; i < num; ++i){
    enc_x_minus[i] = enc_X[i];
    //for slot with value neq 0, minus 8
    //case 0: first line
    if(i == 0){
      Plaintext one;
      encoder.encode(minus,enc_x_minus[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,enc_x_minus[i].parms_id());
      evaluator.sub_plain_inplace(enc_x_minus[i],one);
    }
    //case1: all zero row
    else if(i > input_num && i <= (num-input_num)){

    }
    //case2: 0 - input_num line
    else if(i <= input_num){
      vector<double>temps1(slot_count,0);
      int index = num_batch * (input_num-i);
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i < index){
          temps1[i] = minus_index;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 1;
      Plaintext one;
      encoder.encode(temps1,enc_x_minus[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,enc_x_minus[i].parms_id());
      evaluator.sub_plain_inplace(enc_x_minus[i],one);
    }
    //case3: num-input - num line
    else if(i > num-input_num){
      vector<double>temps1(slot_count,0);
      int index = (num-i) * num_batch;
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i >= index){
          //cout <<i<<endl;
          temps1[i] = minus_index;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 0;
      Plaintext one;
      encoder.encode(temps1,enc_x_minus[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,enc_x_minus[i].parms_id());
      evaluator.sub_plain_inplace(enc_x_minus[i],one);
    }
    //else{
    //  cout <<"ERROR in computing e^x. "<<endl;
   // }

  }



  //compute e^x_ij
  vector<Ciphertext> exp_x(num);
  
  vector<double> s1(slot_count,0);
  for (int i = 0; i < slot_count; ++i){
    if(bias_vec[i] == 1){
      s1[i] = 1;
    }
  }

  #pragma omp parallel for 

  for (int i = 0; i < num; ++i){
    exp_x[i] = exp(enc_x_minus[i],seal_context,relin_keys);

    //for slot with value 0, times 0
    //case 0: first line
    if(i == 0){
      Plaintext one;
      encoder.encode(s1,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.multiply_plain_inplace(exp_x[i],one);
      evaluator.rescale_to_next_inplace(exp_x[i]);
    }
    //case1: all zero row
    else if(i > input_num && i <= (num-input_num)){
      Plaintext one;
      encoder.encode(0,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.multiply_plain_inplace(exp_x[i],one);
      evaluator.rescale_to_next_inplace(exp_x[i]);
    }
    //case2: 0 - input_num line
    else if(i <= input_num){
      vector<double>temps1(slot_count,0);
      int index = num_batch * (input_num-i);
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i < index){
          temps1[i] = 1;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 1;
      Plaintext one;
      encoder.encode(temps1,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.multiply_plain_inplace(exp_x[i],one);
      evaluator.rescale_to_next_inplace(exp_x[i]);
    }
    //case3: num-input - num line
    else if(i > num-input_num){
      vector<double>temps1(slot_count,0);
      int index = (num-i) * num_batch;
      for (int i = 0; i < slot_count; ++i){
        if(bias_vec[i] == 1 && i >= index){
          //cout <<i<<endl;
          temps1[i] = 1;
        }
      }
      //cout <<index/num<<endl;
      //s1[index] = 0;
      Plaintext one;
      encoder.encode(temps1,exp_x[i].scale(),one);
      evaluator.mod_switch_to_inplace(one,exp_x[i].parms_id());
      evaluator.multiply_plain_inplace(exp_x[i],one);
      evaluator.rescale_to_next_inplace(exp_x[i]);
    }
    //else{
    //  cout <<"ERROR in computing e^x. "<<endl;
   // }
    exp_x[i].scale() = scale;

  }

  Plaintext plain_result;
  vector<double> result;
  
  cout <<"Decrypt + decode result of e^(x-13): "<<endl;
    //decrypt and decode
    for (int i = 0; i < 5; ++i){
        Plaintext plain_result;
        decryptor.decrypt(exp_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(bias_vec[ind] == 1){
                cout <<result[ind]<<" ";
            }
        }
        cout <<endl;

    }

    for (int i = exp_x.size()-5; i < exp_x.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(exp_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < slot_count ; ++ind){
            if(bias_vec[ind] == 1){
                cout <<result[ind]<<" ";
            }
        }
        cout <<endl;

    }

  //compute /sum e^x_j
  Ciphertext sum_exp_x = exp_x[0];
  for (int i = 1; i < num; ++i){
    evaluator.add_inplace(sum_exp_x,exp_x[i]);
  }
  //evaluator.rescale_to_next_inplace(sum_exp_x);


  //add 1*10^-5
  Plaintext eps;
  encoder.encode(0.00001, sum_exp_x.parms_id(), sum_exp_x.scale(), eps);
  evaluator.add_plain_inplace(sum_exp_x,eps);
  sum_exp_x.scale()=scale;

  cout <<"  decrypt of sum_exp_(x-13): "<<endl;;
  decryptor.decrypt(sum_exp_x,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
        cout <<"("<<result[ind]<<" "<<1/result[ind]<<") ";
    }
  }
  cout <<endl;

  //mod switch to the lowest level
  while(seal_context.get_context_data(sum_exp_x.parms_id())->chain_index() != 0){
    evaluator.mod_switch_to_next_inplace(sum_exp_x);
  }
  //cout <<"    Modulus chain before bootstrapping: "<< seal_context.get_context_data(sum_exp_x.parms_id())->chain_index()<<endl;

  //bootstrapping sum_exp_(x-8)
  Ciphertext rtn;
  bootstrapper_att.bootstrap_3(rtn,sum_exp_x);
  //cout <<"    Modulus chain after bootstrapping: "<< seal_context.get_context_data(rtn.parms_id())->chain_index()<<endl; 
  while (seal_context.get_context_data(rtn.parms_id())->chain_index() > iter + 1 + 3){
    evaluator.mod_switch_to_next_inplace(rtn);
  }
  //cout <<"    Modulus chain after bootstrapping: "<< seal_context.get_context_data(rtn.parms_id())->chain_index()<<endl;
  //cout <<"    Modulus chain for bootstrapped ct should >= "<<iter+1<<" + modulus chain for e^x"<<endl; 
  //compute Inv(sum_exp_x)
  Ciphertext inv_sum = inverse(rtn,seal_context,relin_keys,iter);
  inv_sum.scale() = scale;
  //cout <<"Modulus chain for inv(sum): "<< seal_context.get_context_data(inv_sum.parms_id())->chain_index()<<endl;
  //cout <<log2(inv_sum.scale())<<endl;
  if(seal_context.get_context_data(exp_x[0].parms_id())->chain_index()
      <seal_context.get_context_data(inv_sum.parms_id())->chain_index()){
      evaluator.mod_switch_to_inplace(inv_sum,exp_x[0].parms_id());
    }
  //cout <<"Modulus chain for modswitch(inv(sum)): "<< seal_context.get_context_data(inv_sum.parms_id())->chain_index()<<endl;
  //cout <<"Modulus chain for modswitch(exp_x): "<< seal_context.get_context_data(exp_x[0].parms_id())->chain_index()<<endl;

  cout <<"  decrypt of inv(sum_exp_(x-8)): "<<endl;
  decryptor.decrypt(inv_sum,plain_result);
  encoder.decode(plain_result,result);
  for (int ind = 0 ; ind < slot_count ; ++ind){
    if(bias_vec[ind] == 1){
       cout <<result[ind]<<" ";
    }
  }
  cout <<endl;

  #pragma omp parallel for 

  for (int i = 0; i < num; ++i){
    if(seal_context.get_context_data(exp_x[i].parms_id())->chain_index()
      >seal_context.get_context_data(inv_sum.parms_id())->chain_index()){
      evaluator.mod_switch_to_inplace(exp_x[i],inv_sum.parms_id());
    }
    evaluator.multiply(exp_x[i],inv_sum,output[i]);
    evaluator.relinearize_inplace(output[i],relin_keys);
    evaluator.rescale_to_next_inplace(output[i]);
    output[i].scale() = scale;
  } 


  return output;

}