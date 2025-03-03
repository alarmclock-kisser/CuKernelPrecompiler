extern "C"
{
    __global__ void StretchInplace(int N, float2* inputArray, float factor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            // Neue Frequenzposition berechnen
            float newIdx = i / factor;
            int lowIdx = (int)floor(newIdx);
            int highIdx = min(lowIdx + 1, N - 1);
            float weight = newIdx - lowIdx;

            // Lineare Interpolation für Frequenzverschiebung
            float2 interpolated;
            interpolated.x = (1.0f - weight) * inputArray[lowIdx].x + weight * inputArray[highIdx].x;
            interpolated.y = (1.0f - weight) * inputArray[lowIdx].y + weight * inputArray[highIdx].y;

            // Synchronisation, falls notwendig
            __syncthreads();

            // Ergebnis in das Input-Array zurückschreiben (In-Place)
            inputArray[i] = interpolated;
        }
    }
}
