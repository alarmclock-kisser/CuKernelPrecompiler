#ifndef M_PI

#define M_PI 3.14159265358979323846f

#endif
extern "C" {
    __global__ void StretchIP_Spline(int size, float2* array, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        float newIdx = i / factor;
        int baseIdx = (int)floorf(newIdx);
        int nextIdx = min(baseIdx + 1, size - 1);

        // Kubische Spline-Interpolation für weiche Übergänge
        float t = newIdx - baseIdx;
        float3 A = {array[max(0, baseIdx - 1)].x, array[baseIdx].x, array[nextIdx].x};
        float3 B = {array[max(0, baseIdx - 1)].y, array[baseIdx].y, array[nextIdx].y};

        float2 interpolated;
        interpolated.x = A.x * (1 - t) + A.y * t;
        interpolated.y = B.x * (1 - t) + B.y * t;

        // Phase-Korrektur
        float phaseA = atan2f(array[baseIdx].y, array[baseIdx].x);
        float phaseB = atan2f(array[nextIdx].y, array[nextIdx].x);
        float phaseDiff = phaseB - phaseA;

        if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
        if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

        float targetPhase = phaseA + t * phaseDiff;
        float magnitude = hypotf(interpolated.x, interpolated.y);

        array[i].x = magnitude * cosf(targetPhase);
        array[i].y = magnitude * sinf(targetPhase);
    }
}