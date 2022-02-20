%%cu

#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <random>

using namespace std;

#define block_size 32


float max_diff(float *res1, float *res2, int n){
    float diff, r = 0;

    for (int i=0; i<n; i++){
        diff = abs(res1[i]-res2[i]);
        r = (r < diff) ? diff : r;
    }

    return r;
}


int n_zeros(float *a, int n){
    int r = 0;

    for (int i=0; i<n; i++){
        r += (!a[i]);
    }
    
    return r;
}


void fill_array(float *a, int n){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> dist(0.0f, 1.0f); 

    for (int i=0; i<n; i++){
        a[i] = dist(gen);
    }
}

void init_zero(float *a, int n){
    for (int i=0; i<n; i++){
        a[i] = 0.0f;
    }
}


void set_eq(float *a, float *b, int n){
    for (int i=0; i<n; i++){
        a[i] = b[i];
    }
}


void kaiming_init(float *w, int n_in, int n_out){
    float std = sqrt(2/(float) n_in);
    
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> dist(0.0f, std); 

    for (int i=0; i<n_in*n_out; i++){
        w[i] = dist(gen);
    }
}


int random_int(int min, int max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

class Module{
    public:
        float *inp, *out;
        int sz_out;
        
        virtual void forward(float *inp, float *out){};
        virtual void backward(){};
        virtual void update(){};
};


class MSE_GPU: public Module{
    public:
        float *inp, *out;
        int n_blocks;
        
        MSE_GPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void _forward(float *_inp, float *_out);
        void backward();
};

class Linear_GPU: public Module{
    public:
        float *weights, *cp_weights, *bias, lr;
        int bs, n_in, n_out, sz_weights, n_block_rows, n_block_cols;

        Linear_GPU(int _bs, int _n_in, int _n_out, float _lr = 0.1f);
        void forward(float *_inp, float *_out);
        void backward();
        void update();

};

class ReLU_GPU: public Module{
    public:
        int n_blocks;
        
        ReLU_GPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void backward();
};

class Sequential_GPU: public Module{
    public:
        std::vector<Module*> layers; 

        Sequential_GPU(std::vector<Module*> _layers);
        void forward(float *inp, float *out);
        void update();
};


__global__
void linear_forward_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        out[ind_out] = bias[col];

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;
            
            out[ind_out] += inp[ind_inp]*weights[ind_weights];
        }
    }
}


__global__
void linear_backward_gpu(float *inp, float *weights, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            atomicAdd(&inp[ind_inp], weights[ind_weights]*out[ind_out]);
        }
    }
}


__global__
void linear_update_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out, float lr){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        atomicAdd(&bias[col], -lr*out[ind_out]);

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            atomicAdd(&weights[ind_weights], -lr*inp[ind_inp]*out[ind_out]);
        }
    }
}


Linear_GPU::Linear_GPU(int _bs, int _n_in, int _n_out, float _lr){
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;

    sz_weights = n_in*n_out;
    sz_out = bs*n_out;
    n_block_rows = (bs + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size - 1) / block_size;

    cudaMallocManaged(&weights, sz_weights*sizeof(float));
    cudaMallocManaged(&bias, n_out*sizeof(float));

    kaiming_init(weights, n_in, n_out);
    init_zero(bias, n_out);
}


void Linear_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_forward_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out);
    cudaDeviceSynchronize();
}


void Linear_GPU::backward(){
    init_zero(inp, bs*n_in);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_backward_gpu<<<n_blocks, n_threads>>>(inp, cp_weights, out, bs, n_in, n_out);
    cudaDeviceSynchronize();

    cudaFree(cp_weights);
}


void Linear_GPU::update(){
    cudaMallocManaged(&cp_weights, sz_weights*sizeof(float));
    set_eq(cp_weights, weights, sz_weights);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);
    
    linear_update_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out, lr);
    cudaDeviceSynchronize();
}



__global__
void mse_forward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        atomicAdd(&out[sz_out], fdividef(powf(inp[ind]-out[ind], 2), sz_out));
    }
}


__global__
void mse_backward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        inp[ind] = fdividef(2*(inp[ind]-out[ind]), sz_out);
    }
}


MSE_GPU::MSE_GPU(int _sz_out){
    sz_out = _sz_out;
    
    n_blocks = (sz_out + block_size - 1) / block_size;
}


void MSE_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;
}


void MSE_GPU::_forward(float *_inp, float *_out){
    _out[sz_out] = 0.0f;
    
    mse_forward_gpu<<<n_blocks, block_size>>>(_inp, _out, sz_out);
    cudaDeviceSynchronize();
}


void MSE_GPU::backward(){
    mse_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}

__global__
void relu_forward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (ind < sz_out){
        out[ind] = fmaxf(0, inp[ind]);
    }
}


__global__
void relu_backward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (ind < sz_out){
        inp[ind] = (0 < inp[ind]) * out[ind];
    }
}


ReLU_GPU::ReLU_GPU(int _sz_out){
    sz_out = _sz_out;
    
    n_blocks = (sz_out + block_size - 1) / block_size;
}


void ReLU_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    relu_forward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}


void ReLU_GPU::backward(){    
    relu_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}

void sequential_forward_gpu(float *inp, std::vector<Module*> layers, float *out){
    int sz_out;
    float *curr_out;

    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];

        sz_out = layer->sz_out;

        cudaMallocManaged(&curr_out, sz_out*sizeof(float));
        layer->forward(inp, curr_out);

        inp = curr_out;
    }

    cudaMallocManaged(&curr_out, sizeof(float));
    cudaFree(curr_out);
}


void sequetial_update_gpu(std::vector<Module*> layers){
    for (int i=layers.size()-1; 0<=i; i--){
        Module *layer = layers[i];

        layer->update(); 
        layer->backward();
    }
}


Sequential_GPU::Sequential_GPU(std::vector<Module*> _layers){
    layers = _layers;
}


void Sequential_GPU::forward(float *inp, float *out){
    sequential_forward_gpu(inp, layers, out);
}


void Sequential_GPU::update(){
    sequetial_update_gpu(layers);
}


void train_gpu(Sequential_GPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs){
    MSE_GPU mse(bs);
    
    int sz_inp = bs*n_in;
    float *cp_inp, *out;
    cudaMallocManaged(&cp_inp, sz_inp*sizeof(float));

    for (int i=0; i<n_epochs; i++){
        set_eq(cp_inp, inp, sz_inp);

        seq.forward(cp_inp, out);
        mse.forward(seq.layers.back()->out, targ);
        
        mse.backward();
        seq.update();
    }
    
    seq.forward(inp, out);
    mse._forward(seq.layers.back()->out, targ);
    std::cout << "La fonction perte est de: " << targ[bs] << std::endl;
}

void read_csv(float *inp, std::string name){
    std::ifstream file(name);
    std::string line;

    while(std::getline(file, line, '\n')){
        *inp = std::stof(line);
        inp++;
    }
}

int main(){
    std::chrono::steady_clock::time_point begin, end;         //To calculate training time

    int bs = 100000, n_in = 50, n_epochs = 100;
    int n_hidden = n_in/2;

    float *inp, *targ;  
    cudaMallocManaged(&inp, bs*n_in*sizeof(float));
    cudaMallocManaged(&targ, (bs+1)*sizeof(float));

    read_csv(inp, "/content/sample_data/x.csv");
    read_csv(targ, "/content/sample_data/x.csv");
    
    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden);
    ReLU_GPU* relu1 = new ReLU_GPU(bs*n_hidden);
    Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden, 1);
    
    std::vector<Module*> layers = {lin1, relu1, lin2};
    Sequential_GPU seq(layers);

    std::cout << "Starting Training : " << std:: endl;
    
    begin = std::chrono::steady_clock::now();         
    train_gpu(seq, inp, targ, bs, n_in, n_epochs);
    end = std::chrono::steady_clock::now();
    std::cout << "Training time with Cuda Gpu: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    return 0;
}
