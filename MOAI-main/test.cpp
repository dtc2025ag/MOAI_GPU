#include <iostream>
#include "include.hpp"

using namespace std;



int main(){

/*
cout <<"----------------------SEAL CKKS BASIC------------------"<<endl;
SEAL_ckks_test();
cout <<endl;
*/


cout <<"----------------------BATCH ENCODE ENCRYPT------------------"<<endl;
batch_input_test();
cout <<endl;



cout <<"----------------------CT-PT MATRIX MULTIPLICATION------------------"<<endl;
ct_pt_matrix_mul_test();
cout <<endl;



cout <<"----------------------CT-CT MATRIX MULTIPLICATION------------------"<<endl;
ct_ct_matrix_mul_test();
cout <<endl;


/*
cout <<"----------------------12 HEAD ATTENTION BLOCK------------------"<<endl;
multi_att_block_test();
cout <<endl;
*/

/*
cout <<"----------------------LAYERNORM FUNCTION + BOOTSTRAPPING------------------"<<endl;
layernorm_bootstrapping_test();
cout <<endl;
*/

/*
cout <<"----------------------SINGLE LAYER------------------"<<endl;
single_layer_test();
cout <<endl;
*/


cout <<"----------------------FULL SCHEME------------------"<<endl;
all_layer_test();
cout <<endl;


  return 0;
}
