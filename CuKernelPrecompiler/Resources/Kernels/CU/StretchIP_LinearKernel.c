#ifndef M_PI

#define M_PI 3.14159265358979323846f

#endif
extern "C" {
    __global__ void StretchIP_Linear(int size, float2* array, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        // Neue Position berechnen
        float newIdx = i / factor;
        int lowIdx = (int)floorf(newIdx);
        int highIdx = min(lowIdx + 1, size - 1);
        float weight = newIdx - lowIdx;

        // Lineare Interpolation
        float2 interpolated;
        interpolated.x = (1.0f - weight) * array[lowIdx].x + weight * array[highIdx].x;
        interpolated.y = (1.0f - weight) * array[lowIdx].y + weight * array[highIdx].y;

        // Phase-Korrektur
        float phaseLow = atan2f(array[lowIdx].y, array[lowIdx].x);
        float phaseHigh = atan2f(array[highIdx].y, array[highIdx].x);
        float phaseDiff = phaseHigh - phaseLow;

        if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
        if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

        float targetPhase = phaseLow + weight * phaseDiff;
        float magnitude = hypotf(interpolated.x, interpolated.y);

        // Update mit korrigierter Phase
        array[i].x = magnitude * cosf(targetPhase);
        array[i].y = magnitude * sinf(targetPhase);
    }
}
