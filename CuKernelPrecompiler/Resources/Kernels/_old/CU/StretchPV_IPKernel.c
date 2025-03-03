#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" {
    __global__ void StretchPV_IP(int n, float2* input, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        // Neue Position im gestreckten Spektrum
        float newIdx = i / factor;
        int lowIdx = (int)floorf(newIdx);
        int highIdx = min(lowIdx + 1, n - 1);
        float weight = newIdx - lowIdx;

        // Interpolierte Amplitude
        float2 interpolated;
        interpolated.x = (1.0f - weight) * input[lowIdx].x + weight * input[highIdx].x;
        interpolated.y = (1.0f - weight) * input[lowIdx].y + weight * input[highIdx].y;
        
        // Magnitude berechnen
        float magnitude = hypotf(interpolated.x, interpolated.y);

        // Phase berechnen
        float phaseLow = atan2f(input[lowIdx].y, input[lowIdx].x);
        float phaseHigh = atan2f(input[highIdx].y, input[highIdx].x);
        
        // Phasenverschiebung Δφ berechnen
        float expectedPhaseDiff = (2.0f * M_PI * i / n) * (n / 2) * factor; // n/2 wegen Overlap
        
        // Tatsächliche Phasendifferenz
        float phaseDiff = phaseHigh - phaseLow;

        // Phasensprünge korrigieren
        if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
        if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

        // Neue Phase berechnen
        float targetPhase = phaseLow + weight * phaseDiff + expectedPhaseDiff;

        // Zurück in Real/Imaginary umwandeln
        input[i].x = magnitude * cosf(targetPhase);
        input[i].y = magnitude * sinf(targetPhase);
    }
}
