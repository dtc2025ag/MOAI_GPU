#include "include.cuh"
// #include "test_ct_pt_matrix_mul.cuh"
// #include "test_phantom_ckks.cuh"
// #include "test_batch_encode_encrypt.cuh"
// #include "test_ct_ct_matrix_mul.cuh"
// #include "test_gelu.cuh"
// #include "test_layernorm.cuh"
// #include "test_softmax.cuh"
// #include "test_single_layer.cuh"
#include <cuda_runtime.h>
using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace moai;

int main()
{

    // cout << "test Phantom ckks" << endl;
    // phantom_ckks_test();
    // cout << "lib test passed!" << endl;

    // cout << "unit test: Batch encode and encrypt" << endl;
    // batch_input_test();
    // cout << "unit test Batch encode and encrypt passed!" << endl;

    // cout << "unit test: Ct-pt matrix multiplication without preprocessing" << endl;
    // ct_pt_matrix_mul_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Ct-pt matrix multiplication with preprocessing" << endl;
    // ct_pt_matrix_mul_w_preprocess_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Ct-ct matrix multiplication" << endl;
    // ct_ct_matrix_mul_test();
    // cout << "unit test Ct-ct matrix multiplication passed!" << endl;

       	cout << "unit test: GeLU" << endl;
    	gelu_test();
    	cout << "unit test passed!" << endl;

    // cout << "unit test: LayerNorm" << endl;
    // layernorm_test();
    // cout << "unit test passed!" << endl;

    // cout << "unit test: Softmax" << endl;
    // softmax_test();
    // cout << "unit test passed!" << endl;

//    cout << "unit test: Bootstrapping" << endl;
//    bootstrapping_test();
//    cout << "unit test passed!" << endl;

    // cout << "unit test: softmax with bootstrapping" << endl;
    // softmax_boot_test();
    // cout << "unit test passed!" << endl;

    // cout << "single layer test" << endl;
    // single_layer_test();
    // cout << "single layer test passed!" << endl;

    return 0;
}
