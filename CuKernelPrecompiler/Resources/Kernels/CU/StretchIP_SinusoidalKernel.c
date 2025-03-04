#ifndef M_PI

#define M_PI 3.14159265358979323846f

#endif
extern "C" {
    __global__ void StretchIP_Sinusoidal(int size, float2* array, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        float newIdx = i / factor;
        int baseIdx = (int)floorf(newIdx);
        int nextIdx = min(baseIdx + 1, size - 1);
        float weight = newIdx - baseIdx;

        // Sinusoidal Amplituden-Interpolation
        float2 interpolated;
        interpolated.x = sinf(weight * M_PI) * array[baseIdx].x + (1 - sinf(weight * M_PI)) * array[nextIdx].x;
        interpolated.y = sinf(weight * M_PI) * array[baseIdx].y + (1 - sinf(weight * M_PI)) * array[nextIdx].y;

        array[i] = interpolated;
    }
}
