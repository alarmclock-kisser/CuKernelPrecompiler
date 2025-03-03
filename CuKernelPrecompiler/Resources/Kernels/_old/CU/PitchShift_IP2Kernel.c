#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" {
    __global__ void PitchShift_IP2(int n, float2* input, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        int overlap = n / 2;  // 50% Overlap

        // Magnitude beibehalten
        float magnitude = hypotf(input[i].x, input[i].y);

        // Originalphase
        float phase = atan2f(input[i].y, input[i].x);

        // Frame-Index innerhalb des Overlap-Bereichs
        int frameIndex = i % overlap;
        float overlapFactor = (float)frameIndex / overlap;

        // **Phasenverschiebung verstärken**, um Pitch-Shift spürbarer zu machen
        float phaseShift = (factor - 1.0f) * 2.0f * M_PI * overlapFactor;

        // Neue Phase berechnen
        float newPhase = phase + phaseShift;

        // Phasensprünge korrigieren
        if (newPhase > M_PI) newPhase -= 2.0f * M_PI;
        if (newPhase < -M_PI) newPhase += 2.0f * M_PI;

        // Zurück in Real/Imaginary umwandeln
        input[i].x = magnitude * cosf(newPhase);
        input[i].y = magnitude * sinf(newPhase);
    }
}
