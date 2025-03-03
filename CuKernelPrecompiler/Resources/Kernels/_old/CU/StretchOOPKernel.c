extern "C"
{
    __global__ void StretchOOP(
        int fftSize,
        const float2* __restrict__ inputFreq,
        float2* __restrict__ outputFreq,
        float stretchFactor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < fftSize)
        {
            // Neue Frequenzposition berechnen
            float newIdx = i * stretchFactor;
            int lowIdx = (int)floor(newIdx);
            int highIdx = min(lowIdx + 1, fftSize - 1);
            float weight = newIdx - lowIdx;

            // Lineare Interpolation für Frequenzverschiebung
            outputFreq[i].x = (1.0f - weight) * inputFreq[lowIdx].x + weight * inputFreq[highIdx].x;
            outputFreq[i].y = (1.0f - weight) * inputFreq[lowIdx].y + weight * inputFreq[highIdx].y;
        }
    }
}
