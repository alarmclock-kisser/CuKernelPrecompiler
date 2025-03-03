#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchIP_Simple_V1(int N, float2* inputArray, float factor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            // Berechne die neue Position (ohne Interpolation - einfach nächster Nachbar)
            int inputIndexInt = (int)roundf((float)i / factor); // Nächster Nachbar Index

            // Stelle sicher, dass der Index im gültigen Bereich ist
            if (inputIndexInt < 0)
                inputIndexInt = 0;
            if (inputIndexInt >= N)
                inputIndexInt = N - 1;

            // Kopiere einfach den Wert des nächsten Nachbarn (keine Interpolation, keine Phasenkorrektur)
            inputArray[i] = inputArray[inputIndexInt];
        }
    }
}