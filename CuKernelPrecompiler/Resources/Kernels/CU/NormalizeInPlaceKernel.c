#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void NormalizeInPlace(int size, float* array, float value)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
        {
            if (value != 0.0f) // Vermeide Division durch Null
            {
                array[i] = array[i] / value; // In-Place Normalisierung: Teile durch 'value'
            }
            // Wenn 'value' 0 ist, lasse den Wert unverändert (oder setze ihn auf 0, je nach gewünschtem Verhalten)
            // Hier wird er unverändert gelassen.
        }
    }
}