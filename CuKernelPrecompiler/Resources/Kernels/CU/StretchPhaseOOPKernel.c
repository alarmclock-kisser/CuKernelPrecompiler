#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C"
{
    __global__ void StretchPhaseOOP(
        int fftSize,
        const float2* __restrict__ inputFreq,
        float2* __restrict__ outputFreq,
        float stretchFactor)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < fftSize)
        {
            // Neue Frequenzposition berechnen
            float newIdx = i * stretchFactor;
            int lowIdx = (int)floorf(newIdx);
            int highIdx = min(lowIdx + 1, fftSize - 1);
            float weight = newIdx - lowIdx;

            // Magnitude und Phase auslesen
            float mag1 = hypotf(inputFreq[lowIdx].x, inputFreq[lowIdx].y);
            float mag2 = hypotf(inputFreq[highIdx].x, inputFreq[highIdx].y);
            float phase1 = atan2f(inputFreq[lowIdx].y, inputFreq[lowIdx].x);
            float phase2 = atan2f(inputFreq[highIdx].y, inputFreq[highIdx].x);

            // Magnitude interpolieren
            float newMag = (1.0f - weight) * mag1 + weight * mag2;

            // Phase korrigieren (Phase Locking)
            float phaseDiff = phase2 - phase1;
            if (phaseDiff > M_PI) phaseDiff -= 2 * M_PI;
            if (phaseDiff < -M_PI) phaseDiff += 2 * M_PI;
            float newPhase = phase1 + weight * phaseDiff;

            // Hanning-Windowing anwenden
            float window = 0.5f * (1 - cosf(2.0f * M_PI * i / (fftSize - 1)));
            newMag *= window;

            // Rückumwandlung zu Real- und Imaginärteil
            outputFreq[i].x = newMag * cosf(newPhase);
            outputFreq[i].y = newMag * sinf(newPhase);
        }
    }
}
