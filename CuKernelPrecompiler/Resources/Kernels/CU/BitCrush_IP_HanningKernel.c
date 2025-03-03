#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#include <math.h>

extern "C"
{
    __global__ void BitCrush_IP_Hanning(int N, float* array, float factor)
    {
        int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int outputN = (int)(N); // Output-Größe ist gleich Input-Größe für In-Place

        if (outputIndex < outputN)
        {
            float sample = array[outputIndex]; // Sample-Wert holen

            // **Bitcrushing Effekt**
            float quantizationFactor = max(1.0f, factor); // Stelle sicher, dass quantizationFactor >= 1 ist
            int quantizationLevels = max(2, (int)(256.0f / quantizationFactor)); // Anzahl der Quantisierungsstufen, skaliert mit Faktor
            float quantizationStep = 2.0f / quantizationLevels; // Schrittgröße für Quantisierung
            float quantizedSample = roundf(sample / quantizationStep) * quantizationStep; // Quantisierung

            // Stelle sicher, dass der quantisierte Wert im Bereich [-1, 1] bleibt (optional, aber sicherheitshalber)
            quantizedSample = fminf(1.0f, fmaxf(-1.0f, quantizedSample));

            // **Hanning-Fensterfunktion IN-PLACE anwenden** (für N/2 Overlap)
            float windowValue = 1.0f; // Standardwert (kein Fenster)
            if (factor > 0.0f) // Fenster nur anwenden, wenn Effekt aktiv ist (optional)
            {
                windowValue = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)outputIndex / (float)outputN)); // Hanning-Fenster
            }

            array[outputIndex] = quantizedSample * windowValue; // In-Place: Quantisierten und gefensterten Wert zurückschreiben
        }
    }
}