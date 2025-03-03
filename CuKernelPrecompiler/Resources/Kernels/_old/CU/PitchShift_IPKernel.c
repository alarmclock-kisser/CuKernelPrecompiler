#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" {
    __global__ void PitchShift_IP(int n, float2* input, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        int overlap = n / 2;  // 50% Overlap

        // Magnitude beibehalten
        float magnitude = hypotf(input[i].x, input[i].y);

        // Phase extrahieren
        float phase = atan2f(input[i].y, input[i].x);

        // Overlap-Korrektur: Phasen überlappt anpassen
        int frameIndex = i % overlap;
        float overlapFactor = (float)frameIndex / overlap;

        // Phase nach Pitch-Faktor anpassen
        float newPhase = phase * factor + overlapFactor * M_PI;

        // Phasensprünge korrigieren
        if (newPhase > M_PI) newPhase -= 2.0f * M_PI;
        if (newPhase < -M_PI) newPhase += 2.0f * M_PI;

        // Zurück in Real/Imaginary umwandeln
        input[i].x = magnitude * cosf(newPhase);
        input[i].y = magnitude * sinf(newPhase);
    }
}
