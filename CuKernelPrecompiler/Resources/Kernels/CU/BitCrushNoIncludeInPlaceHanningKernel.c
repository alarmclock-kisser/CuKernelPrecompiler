#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void BitCrushNoIncludeInPlaceHanning(int N, float* array, float factor)
    {
        int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int outputN = (int)(N); // Output-Größe ist gleich Input-Größe für In-Place

        if (outputIndex < outputN)
        {
            float sample = array[outputIndex]; // Sample-Wert holen

            // **Bitcrushing Effekt - OHNE includes, mit CUDA Built-ins**
            float quantizationFactor = fmaxf(1.0f, factor); // CUDA built-in fmaxf anstelle von std::max
            int quantizationLevels = max(2, (int)(256.0f / quantizationFactor)); // Verwende hier Standard-C++ max (für int), das oft ohne Include funktioniert
            float quantizationStep = 2.0f / quantizationLevels;
            float quantizedSample = roundf(sample / quantizationStep) * quantizationStep; // CUDA built-in roundf anstelle von std::round

            // Stelle sicher, dass der quantisierte Wert im Bereich [-1, 1] bleibt (optional, aber sicherheitshalber)
            quantizedSample = fminf(1.0f, fmaxf(-1.0f, quantizedSample)); // CUDA built-in fminf und fmaxf

            // **Hanning-Fensterfunktion IN-PLACE anwenden - OHNE includes, mit CUDA Built-in cosf** (für N/2 Overlap)
            float windowValue = 1.0f; // Standardwert (kein Fenster)
            if (factor > 0.0f) // Fenster nur anwenden, wenn Effekt aktiv ist (optional)
            {
                windowValue = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)outputIndex / (float)outputN)); // CUDA built-in cosf anstelle von std::cos
            }

            array[outputIndex] = quantizedSample * windowValue; // In-Place: Quantisierten und gefensterten Wert zurückschreiben
        }
    }
}