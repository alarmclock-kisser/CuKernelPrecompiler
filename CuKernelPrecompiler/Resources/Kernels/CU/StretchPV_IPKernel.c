#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchPV_IP(int N, float2* input, float factor, float2* prevPhase)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        // Neue Position im gestreckten Signal
        float newIdx = i / factor;
        int lowIdx = (int)floorf(newIdx);
        int highIdx = min(lowIdx + 1, N - 1);
        float weight = newIdx - lowIdx;

        // Interpolierte Amplitude
        float2 interpolated;
        interpolated.x = (1.0f - weight) * input[lowIdx].x + weight * input[highIdx].x;
        interpolated.y = (1.0f - weight) * input[lowIdx].y + weight * input[highIdx].y;

        // Phase berechnen
        float phaseLow = atan2f(input[lowIdx].y, input[lowIdx].x);
        float phaseHigh = atan2f(input[highIdx].y, input[highIdx].x);
        float phaseDiff = phaseHigh - phaseLow;

        // Phasensprung minimieren
        if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
        if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

        // Zielphase bestimmen
        float targetPhase = phaseLow + weight * phaseDiff;

        // Korrektur der globalen Phasendrift
        float prevPhaseValue = atan2f(prevPhase[i].y, prevPhase[i].x);
        float phaseCorrection = targetPhase - prevPhaseValue;

        if (phaseCorrection > M_PI) phaseCorrection -= 2.0f * M_PI;
        if (phaseCorrection < -M_PI) phaseCorrection += 2.0f * M_PI;

        targetPhase = prevPhaseValue + phaseCorrection;
        float magnitude = hypotf(interpolated.x, interpolated.y);

        // Neue Werte berechnen
        interpolated.x = magnitude * cosf(targetPhase);
        interpolated.y = magnitude * sinf(targetPhase);

        __syncthreads();

        // In-Place Update
        input[i] = interpolated;
        prevPhase[i] = interpolated; // Speichert die aktuelle Phase für das nächste Fenster
    }
}
