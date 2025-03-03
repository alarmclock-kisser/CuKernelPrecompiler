#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchIP_PhaseLock(int N, float2* inputArray, float factor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        // Neue Position im skalierten Frequenzbereich
        float newIdx = i / factor;
        int lowIdx = (int)floorf(newIdx);
        int highIdx = min(lowIdx + 1, N - 1);
        float weight = newIdx - lowIdx;

        // Amplitudeninterpolation
        float2 lowVal = inputArray[lowIdx];
        float2 highVal = inputArray[highIdx];

        float magLow = hypotf(lowVal.x, lowVal.y);
        float magHigh = hypotf(highVal.x, highVal.y);
        float magnitude = (1.0f - weight) * magLow + weight * magHigh;

        // Phase mit Phase-Locking korrigieren
        float phaseLow = atan2f(lowVal.y, lowVal.x);
        float phaseHigh = atan2f(highVal.y, highVal.x);
        float phaseDiff = phaseHigh - phaseLow;

        if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
        if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

        float phaseCorrection = phaseLow + weight * phaseDiff;
        
        // Wiederherstellen des Signals mit korrigierter Phase
        float2 stretchedValue;
        stretchedValue.x = magnitude * cosf(phaseCorrection);
        stretchedValue.y = magnitude * sinf(phaseCorrection);

        __syncthreads();

        // In-Place Update
        inputArray[i] = stretchedValue;
    }
}
