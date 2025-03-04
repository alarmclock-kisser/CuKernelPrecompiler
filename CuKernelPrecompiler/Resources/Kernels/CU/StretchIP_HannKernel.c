#ifndef M_PI

#define M_PI 3.14159265358979323846f

#endif
extern "C" {
    __global__ void StretchIP_Hann(int size, float2* array, float factor) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size) return;

        float newIdx = i / factor;
        int baseIdx = (int)floorf(newIdx);
        int nextIdx = min(baseIdx + 1, size - 1);
        float weight = newIdx - baseIdx;

        // Hann-Window Overlap-Add
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * weight));
        float2 interpolated;
        interpolated.x = window * array[baseIdx].x + (1 - window) * array[nextIdx].x;
        interpolated.y = window * array[baseIdx].y + (1 - window) * array[nextIdx].y;

        array[i] = interpolated;
    }
}
