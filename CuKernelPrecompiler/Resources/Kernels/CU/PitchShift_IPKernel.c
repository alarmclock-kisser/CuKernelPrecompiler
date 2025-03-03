#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void PitchShift_IP(int size, float2* array, float factor)
    {
        int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (outputIndex < size)
        {
            // Berechne den entsprechenden Index im Eingangsarray für die Tonhöhenverschiebung
            float inputIndexFloat = (float)outputIndex * factor; // Multiplikation für Pitch Shift (Frequenzskalierung)
            int inputIndexInt = (int)floorf(inputIndexFloat);

            if (inputIndexInt >= 0 && inputIndexInt < size)
            {
                // Kopiere den Frequenzwert vom skalierten Index (Pitch Shift)
                array[outputIndex] = array[inputIndexInt]; // In-Place: Schreibe direkt in das Input Array
            }
            else
            {
                // Index außerhalb der Grenzen: Setze auf 0 (Stille/keine Frequenzkomponente)
                array[outputIndex] = make_float2(0.0f, 0.0f);
            }
        }
    }
}