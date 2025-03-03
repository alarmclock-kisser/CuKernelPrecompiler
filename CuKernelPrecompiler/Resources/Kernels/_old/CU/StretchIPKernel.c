#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchIP(int N, float2* inputArray, float factor, int samples)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            // Berechne die neue Position mit Overlap
            float newIdx = (i / factor) - (samples / factor);
            int lowIdx = max(0, (int)floor(newIdx));
            int highIdx = min(lowIdx + 1, N - 1);
            float weight = newIdx - lowIdx;

            // Frequenzamplituden interpolieren
            float2 interpolated;
            interpolated.x = (1.0f - weight) * inputArray[lowIdx].x + weight * inputArray[highIdx].x;
            interpolated.y = (1.0f - weight) * inputArray[lowIdx].y + weight * inputArray[highIdx].y;

            // ---- PHASENKORREKTUR ----
            float phaseDiff = atan2f(inputArray[highIdx].y, inputArray[highIdx].x) - 
                              atan2f(inputArray[lowIdx].y, inputArray[lowIdx].x);

            // Korrektur, falls Phasensprung zu groß ist
            if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
            if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

            float targetPhase = atan2f(inputArray[lowIdx].y, inputArray[lowIdx].x) + weight * phaseDiff;

            // Phase und Amplitude kombinieren
            float magnitude = hypotf(interpolated.x, interpolated.y);
            interpolated.x = magnitude * cosf(targetPhase);
            interpolated.y = magnitude * sinf(targetPhase);

            // Synchronisation für sicheres Schreiben
            __syncthreads();

            // Ergebnis zurückschreiben (In-Place)
            inputArray[i] = interpolated;
        }
    }
}
