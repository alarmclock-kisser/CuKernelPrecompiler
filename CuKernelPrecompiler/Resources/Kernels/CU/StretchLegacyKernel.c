extern "C"
{
    __global__ void StretchLegacy(const int N, float2* __restrict data, float factor)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N)
        {
            // Berechnung der Magnitude und Phase
            float magnitude = sqrtf(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
            float phase = atan2f(data[idx].y, data[idx].x);

            // Stretching der Phase
            phase *= factor;

            // Rekonstruktion des komplexen Wertes
            data[idx].x = magnitude * cosf(phase);
            data[idx].y = magnitude * sinf(phase);
        }
    }
}