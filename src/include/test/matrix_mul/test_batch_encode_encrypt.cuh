using namespace std;
using namespace seal;

void batch_input_test(){
    cout <<"Task: test batch input + encoding + encrypting in CKKS scheme: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0,40);

    SEALContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);

    struct timeval tstart1, tend1;

    //construct input
    int num_X = 256;
    int num_row = 128;
    int num_col = 768;
    cout <<"Number of matrices in one batch = "<<num_X<<endl;
    vector<vector<vector<double>>> input_x(num_X,vector<vector<double>>(num_row, vector<double>(num_col,0)));
    for (int i = 0; i < num_X; ++i){
        for (int j = 0 ; j < num_row ; ++j){
            for (int k = 0 ; k < num_col ; ++k){
                input_x[i][j][k] = (double)j+1.0;
            }
            /*
            for (int k = 0; k < 10; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<"... ";
            for (int k = num_col-10 ; k<num_col ; ++k){
                cout <<input_x[i][j][k]<<" ";
            }
            cout <<endl;
            */
        }

    }
    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<endl;
    

    //encode + encrypt
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context,public_key);

    gettimeofday(&tend1,NULL);
    double enc_ecd_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"encode + encrypt time = "<<enc_ecd_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;


    cout <<"Decrypt + decode result: "<<endl;
    //decrypt
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    for (int i = 0; i < 5; ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
    cout << "......"<<endl;
    for (int i = num_col-5; i < num_col; ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i+1<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-5 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
}








