#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchWSOLA(int N, float2* inputArray, float factor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int halfN = N / 2;  // 50% Overlap-Size

        if (i < N)
        {
            // Berechneter Index für Frequenzverschiebung
            float newIdx = i / factor;
            int lowIdx = (int)floor(newIdx);
            int highIdx = min(lowIdx + 1, N - 1);
            float weight = newIdx - lowIdx;

            // Frequenzinterpolation (Amplitude)
            float2 interpolated;
            interpolated.x = (1.0f - weight) * inputArray[lowIdx].x + weight * inputArray[highIdx].x;
            interpolated.y = (1.0f - weight) * inputArray[lowIdx].y + weight * inputArray[highIdx].y;

            // PHASENKORREKTUR für Overlap-Add
            float phaseLow = atan2f(inputArray[lowIdx].y, inputArray[lowIdx].x);
            float phaseHigh = atan2f(inputArray[highIdx].y, inputArray[highIdx].x);
            float phaseDiff = phaseHigh - phaseLow;

            // Phasensprünge korrigieren
            if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
            if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

            // Zielphase berechnen mit Phase-Locked Adjustment
            float targetPhase = phaseLow + weight * phaseDiff;
            float magnitude = hypotf(interpolated.x, interpolated.y);
            interpolated.x = magnitude * cosf(targetPhase);
            interpolated.y = magnitude * sinf(targetPhase);

            __syncthreads(); // Synchronisation für parallelen Zugriff

            // **Speicherung mit 50% Overlap-Add**
            if (i < halfN)
            {
                inputArray[i] = interpolated;
            }
            else
            {
                // **Weiches Crossfade für Overlap-Region**
                float fadeFactor = (i - halfN) / (float)halfN;  // Werte zwischen 0 und 1
                inputArray[i].x = (1.0f - fadeFactor) * inputArray[i].x + fadeFactor * interpolated.x;
                inputArray[i].y = (1.0f - fadeFactor) * inputArray[i].y + fadeFactor * interpolated.y;
            }
        }
    }
}
