#ifndef M_PI

#define M_PI 3.14159265358979323846f

#endif

extern "C"
{
    __global__ void LiveStretchingOOP(
        int inputFFTSize,
        int outputFFTSize,
        float2* __restrict__ inputFreq,
        float2* __restrict__ outputFreq,
        float stretchFactor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < outputFFTSize)
        {
            // 1. Berechne die entsprechende Position im Input-Array (für Zeitstreckung)
            float inputIndexFloat = (float)i / stretchFactor;
            int inputIndexInt = (int)floorf(inputIndexFloat);
            float fraction = inputIndexFloat - inputIndexInt;

            // 2. Lineare Interpolation der FREQUENZAMPLITUDEN (Magnitude)
            float2 interpolatedMagnitude;
            if (inputIndexInt >= 0 && inputIndexInt < inputFFTSize - 1)
            {
                interpolatedMagnitude.x = (1.0f - fraction) * hypotf(inputFreq[inputIndexInt].x, inputFreq[inputIndexInt].y) +
                                        fraction * hypotf(inputFreq[inputIndexInt + 1].x, inputFreq[inputIndexInt + 1].y); // Interpolation der Magnituden (Realteil ist Magnitude)
                interpolatedMagnitude.y = 0.0f; // Imaginärteil für Magnitude Interpolation nicht relevant, auf 0 setzen
            }
            else if (inputIndexInt >= 0 && inputIndexInt < inputFFTSize)
            {
                interpolatedMagnitude.x = hypotf(inputFreq[inputIndexInt].x, inputFreq[inputIndexInt].y);
                interpolatedMagnitude.y = 0.0f;
            }
            else
            {
                interpolatedMagnitude = make_float2(0.0f, 0.0f);
            }

            // 3. PHASENKORREKTUR (WICHTIG gegen Pitch-Shift!)
            float targetPhase;
            if (inputIndexInt >= 0 && inputIndexInt < inputFFTSize - 1)
            {
                 float phaseDiff = atan2f(inputFreq[inputIndexInt + 1].y, inputFreq[inputIndexInt + 1].x) -
                                   atan2f(inputFreq[inputIndexInt].y, inputFreq[inputIndexInt].x);

                // Phasensprung Korrektur (wie in StretchPhaseOOP)
                if (phaseDiff > M_PI) phaseDiff -= 2.0f * M_PI;
                if (phaseDiff < -M_PI) phaseDiff += 2.0f * M_PI;

                targetPhase = atan2f(inputFreq[inputIndexInt].y, inputFreq[inputIndexInt].x) + fraction * phaseDiff;
            }
            else if (inputIndexInt >= 0 && inputIndexInt < inputFFTSize)
            {
                targetPhase = atan2f(inputFreq[inputIndexInt].y, inputFreq[inputIndexInt].x); // Randfall: Phase einfach übernehmen
            }
            else
            {
                targetPhase = 0.0f; // Fallback Phase: 0
            }


            // 4. Kombiniere interpolierte Magnitude und korrigierte Phase
            outputFreq[i].x = interpolatedMagnitude.x * cosf(targetPhase); // Realteil: Magnitude * cos(Phase)
            outputFreq[i].y = interpolatedMagnitude.x * sinf(targetPhase); // Imaginärteil: Magnitude * sin(Phase)
        }
    }
}