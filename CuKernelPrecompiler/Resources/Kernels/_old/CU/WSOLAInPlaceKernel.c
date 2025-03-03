#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void WSOLAInPlace(int N, float* array, float factor)
    {
        int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int outputN = (int)(N * factor); // Berechne outputN innerhalb des Kernels basierend auf N und factor

        if (outputIndex < outputN)
        {
            float inputIndexFloat = (float)outputIndex / factor;
            int inputIndexInt = (int)floorf(inputIndexFloat);
            float fraction = inputIndexFloat - inputIndexInt;

            if (inputIndexInt >= 0 && inputIndexInt < N - 1)
            {
                // Lineare Interpolation für WSOLA
                float val1 = array[inputIndexInt];
                float val2 = array[inputIndexInt + 1];
                array[outputIndex] = val1 + fraction * (val2 - val1); // **In-Place: Schreibe direkt in das Input Array (array)**
            }
            else if (inputIndexInt >= 0 && inputIndexInt < N)
            {
                array[outputIndex] = array[inputIndexInt]; // Randfall
            }
            else
            {
                array[outputIndex] = 0.0f; // Fallback
            }
        }
    }
}