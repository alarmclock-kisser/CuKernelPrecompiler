#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void WSOLA_IP_Hanning(int N, float* array, float factor)
    {
        int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int outputN = (int)(N * factor); // Berechne outputN innerhalb des Kernels

        if (outputIndex < outputN)
        {
            float inputIndexFloat = (float)outputIndex / factor;
            int inputIndexInt = (int)floorf(inputIndexFloat);
            float fraction = inputIndexFloat - inputIndexInt;

            float interpolatedValue = 0.0f;

            if (inputIndexInt >= 0 && inputIndexInt < N - 1)
            {
                // Lineare Interpolation für WSOLA
                float val1 = array[inputIndexInt];
                float val2 = array[inputIndexInt + 1];
                interpolatedValue = val1 + fraction * (val2 - val1);
            }
            else if (inputIndexInt >= 0 && inputIndexInt < N)
            {
                interpolatedValue = array[inputIndexInt]; // Randfall
            }
            else
            {
                interpolatedValue = 0.0f; // Fallback
            }

            // **Hanning-Fensterfunktion IN-PLACE anwenden**
            float windowValue = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)outputIndex / (float)outputN)); // Hanning-Fenster
            array[outputIndex] = interpolatedValue * windowValue; // In-Place: Fenster anwenden und direkt in 'array' schreiben
        }
    }
}