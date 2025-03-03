#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchIP_Linear(int N, float2* inputArray, float factor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        // Neue Position im skalierten Frequenzbereich
        float newIdx = i / factor;
        int lowIdx = (int)floorf(newIdx);
        int highIdx = min(lowIdx + 1, N - 1);
        float weight = newIdx - lowIdx;

        // Interpolierte Amplitude
        float2 interpolated;
        interpolated.x = (1.0f - weight) * inputArray[lowIdx].x + weight * inputArray[highIdx].x;
        interpolated.y = (1.0f - weight) * inputArray[lowIdx].y + weight * inputArray[highIdx].y;

        // Phase berechnen und korrigieren
        float phaseLow = atan2f(inputArray[lowIdx].y, inputArray[lowIdx].x);
        float phaseHigh = atan2f(inputArray[highIdx].y, inputArray[highIdx].x);
        float phaseDiff = phaseHigh - phaseLow;

        // Phasensprung minimieren
        if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
        if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

        float targetPhase = phaseLow + weight * phaseDiff;
        float magnitude = hypotf(interpolated.x, interpolated.y);

        // Phase und Amplitude kombinieren
        interpolated.x = magnitude * cosf(targetPhase);
        interpolated.y = magnitude * sinf(targetPhase);

        __syncthreads();

        // In-Place Update
        inputArray[i] = interpolated;
    }
}
