#ifndef M_PI

#define M_PI 3.14159265358979323846f

#endif
extern "C" {
    __global__ void StretchIP_PhaseVocoder(int size, float2* array, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        float newIdx = i / factor;
        int baseIdx = (int)floorf(newIdx);
        int nextIdx = min(baseIdx + 1, size - 1);

        float weight = newIdx - baseIdx;

        // Amplitude-Interpolation
        float2 interpolated;
        interpolated.x = (1.0f - weight) * array[baseIdx].x + weight * array[nextIdx].x;
        interpolated.y = (1.0f - weight) * array[baseIdx].y + weight * array[nextIdx].y;

        // Phase-Korrektur mit Differenz zur vorherigen Phase
        float phaseA = atan2f(array[baseIdx].y, array[baseIdx].x);
        float phaseB = atan2f(array[nextIdx].y, array[nextIdx].x);
        float phaseDiff = fmodf(phaseB - phaseA, 2.0f * M_PI);

        if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
        if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

        float targetPhase = phaseA + weight * phaseDiff;
        float magnitude = hypotf(interpolated.x, interpolated.y);

        array[i].x = magnitude * cosf(targetPhase);
        array[i].y = magnitude * sinf(targetPhase);
    }
}
