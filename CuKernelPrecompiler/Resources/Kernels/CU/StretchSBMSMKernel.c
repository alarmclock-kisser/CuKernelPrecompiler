#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchSBMSM(int N, float2* inputArray, float factor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            // Skalierter Frequenzindex
            float newIdx = i / factor;
            int lowIdx = (int)floor(newIdx);
            int highIdx = min(lowIdx + 1, N - 1);
            float weight = newIdx - lowIdx;

            // Frequenzinterpolation (Amplitude)
            float2 interpolated;
            interpolated.x = (1.0f - weight) * inputArray[lowIdx].x + weight * inputArray[highIdx].x;
            interpolated.y = (1.0f - weight) * inputArray[lowIdx].y + weight * inputArray[highIdx].y;

            // Phasenkorrektur
            float phaseLow = atan2f(inputArray[lowIdx].y, inputArray[lowIdx].x);
            float phaseHigh = atan2f(inputArray[highIdx].y, inputArray[highIdx].x);
            float phaseDiff = phaseHigh - phaseLow;

            // Phasensprünge korrigieren
            if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
            if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

            float targetPhase = phaseLow + weight * phaseDiff;

            // Neue Werte mit korrekter Phase
            float magnitude = hypotf(interpolated.x, interpolated.y);
            interpolated.x = magnitude * cosf(targetPhase);
            interpolated.y = magnitude * sinf(targetPhase);

            __syncthreads(); // Synchronisation für parallelen Zugriff

            // Ergebnis speichern
            inputArray[i] = interpolated;
        }
    }
}
