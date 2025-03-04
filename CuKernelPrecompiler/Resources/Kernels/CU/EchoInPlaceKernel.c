#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void EchoInPlace(int size, float* array, float value)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < size)
        {
            float originalSample = array[index];
            float delayedSample = 0.0f;
            int delaySamples = (int)(10000 * value); // Delay-Zeit in Samples, skaliert mit 'value' (0-1)

            if (delaySamples > 0 && index >= delaySamples)
            {
                delayedSample = array[index - delaySamples]; // Sample vom verzögerten Zeitpunkt holen
            }

            array[index] = originalSample + delayedSample * value; // Mische Original und verzögertes Sample, 'value' steuert Feedback/Echo-Stärke
        }
    }
}