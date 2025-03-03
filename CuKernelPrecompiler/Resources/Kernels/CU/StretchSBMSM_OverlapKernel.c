#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchSBMSM_Overlap(int N, float2* inputArray, float factor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int halfN = N / 2; // 50% Overlap

        if (i < N)
        {
            // Berechneter Index unter Berücksichtigung des Overlaps
            float newIdx = i / factor;
            int lowIdx = (int)floor(newIdx);
            int highIdx = min(lowIdx + 1, N - 1);
            float weight = newIdx - lowIdx;

            // Frequenzinterpolation (Amplitude)
            float2 interpolated;
            interpolated.x = (1.0f - weight) * inputArray[lowIdx].x + weight * inputArray[highIdx].x;
            interpolated.y = (1.0f - weight) * inputArray[lowIdx].y + weight * inputArray[highIdx].y;

            // Phasenkorrektur für das Overlap
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

            // **Speicherung mit 50% Overlap Handling**
            if (i < halfN)
            {
                // Erste Hälfte normal überschreiben
                inputArray[i] = interpolated;
            }
            else
            {
                // Zweite Hälfte **additiv** mischen für Overlap
                inputArray[i].x = (inputArray[i].x + interpolated.x) * 0.5f;
                inputArray[i].y = (inputArray[i].y + interpolated.y) * 0.5f;
            }
        }
    }
}
