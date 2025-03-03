extern "C"
{
    __global__ void StretchX(
        int fftSize,
        float2* __restrict__ freqData,
        float stretchFactor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float2 temp[1024]; // Temporärer Speicher (nur für kleine FFTs sinnvoll)

        if (i < fftSize)
        {
            // Neue Frequenzposition
            float newIdx = i * stretchFactor;
            int lowIdx = (int)floor(newIdx);
            int highIdx = min(lowIdx + 1, fftSize - 1);
            float weight = newIdx - lowIdx;

            // Temporären Speicher füllen
            temp[i] = freqData[i];

            __syncthreads(); // Warten auf alle Threads

            // Lineare Interpolation zwischen benachbarten Frequenzen
            freqData[i].x = (1.0f - weight) * temp[lowIdx].x + weight * temp[highIdx].x;
            freqData[i].y = (1.0f - weight) * temp[lowIdx].y + weight * temp[highIdx].y;
        }
    }
}
