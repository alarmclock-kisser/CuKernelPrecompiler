extern "C"
{
    __global__ void Normalize(const int N, float* __restrict data, float maxAmplitude)
    {
        // grid-stride loop
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            data[i] = data[i] * maxAmplitude;
        }
    }
}