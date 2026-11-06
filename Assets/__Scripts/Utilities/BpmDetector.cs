using System;
using UnityEngine;

namespace ChroMapper.BpmDetection
{
    /// <summary>
    ///     Lightweight, dependency-free BPM detector that runs entirely on the local machine.
    ///     Uses spectral-flux onset detection followed by autocorrelation and octave correction.
    /// </summary>
    public static class BpmDetector
    {
        public const float MinBpm = 60f;
        public const float MaxBpm = 200f;
        public const float BpmStep = 0.5f;

        private const int FrameSize = 512;
        private const int HopSize = 256;

        public static float Detect(AudioClip clip)
        {
            if (clip == null || clip.samples <= 0 || clip.frequency <= 0)
                return 0f;

            var channels = clip.channels;
            var sampleCount = clip.samples;
            var data = new float[sampleCount * channels];
            if (!clip.GetData(data, 0))
                return 0f;

            var mono = new float[sampleCount];
            if (channels == 1)
            {
                Array.Copy(data, mono, sampleCount);
            }
            else
            {
                for (var i = 0; i < sampleCount; i++)
                {
                    var sum = 0f;
                    for (var c = 0; c < channels; c++)
                        sum += data[i * channels + c];
                    mono[i] = sum / channels;
                }
            }

            return Detect(mono, clip.frequency);
        }

        public static float Detect(float[] samples, int sampleRate)
        {
            if (samples == null || samples.Length < sampleRate || sampleRate <= 0)
                return 0f;

            var frameCount = (samples.Length - FrameSize) / HopSize + 1;
            if (frameCount < 8)
                return 0f;

            var onset = ComputeOnset(samples, sampleRate, frameCount);
            if (onset == null)
                return 0f;

            // Trim a little of the intro/outro, which often lacks a clear beat.
            var trim = (int)(onset.Length * 0.15f);
            if (trim > 0 && onset.Length - 2 * trim >= 8)
            {
                var trimmed = new float[onset.Length - 2 * trim];
                Array.Copy(onset, trim, trimmed, 0, trimmed.Length);
                onset = trimmed;
                frameCount = onset.Length;
            }

            var framesPerSecond = sampleRate / (float)HopSize;

            // Autocorrelation over candidate tempos.
            float bestScore = 0f;
            float bestBpm = 120f;
            for (var bpm = MinBpm; bpm <= MaxBpm; bpm += BpmStep)
            {
                var lag = (int)Math.Round(framesPerSecond * 60f / bpm);
                if (lag < 1 || lag >= frameCount)
                    continue;

                var score = 0f;
                for (var i = 0; i + lag < frameCount; i++)
                    score += onset[i] * onset[i + lag];

                if (score > bestScore)
                {
                    bestScore = score;
                    bestBpm = bpm;
                }
            }

            if (bestScore <= 0f)
                return 0f;

            // Octave correction only at the extremes, to avoid over-correcting mid-tempo songs.
            var currentBpm = bestBpm;
            var currentScore = bestScore;

            if (currentBpm < 90f && currentBpm * 2f <= MaxBpm)
            {
                var doubled = currentBpm * 2f;
                var lag = (int)Math.Round(framesPerSecond * 60f / doubled);
                if (lag >= 1 && lag < frameCount)
                {
                    var score = 0f;
                    for (var i = 0; i + lag < frameCount; i++)
                        score += onset[i] * onset[i + lag];
                    if (score >= currentScore * 0.95f)
                    {
                        currentBpm = doubled;
                        currentScore = score;
                    }
                }
            }

            if (currentBpm > 180f && currentBpm / 2f >= MinBpm)
            {
                var halved = currentBpm / 2f;
                var lag = (int)Math.Round(framesPerSecond * 60f / halved);
                if (lag >= 1 && lag < frameCount)
                {
                    var score = 0f;
                    for (var i = 0; i + lag < frameCount; i++)
                        score += onset[i] * onset[i + lag];
                    if (score >= currentScore * 0.95f)
                    {
                        currentBpm = halved;
                        currentScore = score;
                    }
                }
            }

            return (float)Math.Round(currentBpm, 1);
        }

        private static float[] ComputeOnset(float[] samples, int sampleRate, int frameCount)
        {
            var hann = new float[FrameSize];
            for (var i = 0; i < FrameSize; i++)
                hann[i] = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (FrameSize - 1)));

            var prevMag = new float[FrameSize / 2 + 1];
            var flux = new float[frameCount];
            var re = new float[FrameSize];
            var im = new float[FrameSize];

            for (var f = 0; f < frameCount; f++)
            {
                var offset = f * HopSize;
                for (var i = 0; i < FrameSize; i++)
                {
                    var idx = offset + i;
                    var v = idx < samples.Length ? samples[idx] : 0f;
                    re[i] = v * hann[i];
                    im[i] = 0f;
                }

                Fft(re, im);

                // spectral flux
                if (f > 0)
                {
                    var frameFlux = 0f;
                    for (var i = 0; i <= FrameSize / 2; i++)
                    {
                        var mag = Mathf.Sqrt(re[i] * re[i] + im[i] * im[i]);
                        var diff = mag - prevMag[i];
                        if (diff > 0f) frameFlux += diff;
                        prevMag[i] = mag;
                    }
                    flux[f] = frameFlux;
                }
                else
                {
                    for (var i = 0; i <= FrameSize / 2; i++)
                        prevMag[i] = Mathf.Sqrt(re[i] * re[i] + im[i] * im[i]);
                    flux[f] = 0f;
                }
            }

            // 3-point smoothing
            var onset = new float[frameCount];
            for (var f = 1; f < frameCount - 1; f++)
                onset[f] = (flux[f - 1] + flux[f] + flux[f + 1]) / 3f;
            onset[0] = flux[0];
            onset[frameCount - 1] = flux[frameCount - 1];

            return onset;
        }

        private static void Fft(float[] re, float[] im)
        {
            var n = re.Length;

            // Bit-reversal permutation
            for (int i = 1, j = 0; i < n; i++)
            {
                var bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1)
                    j ^= bit;
                j ^= bit;
                if (i < j)
                {
                    var tr = re[i]; re[i] = re[j]; re[j] = tr;
                    var ti = im[i]; im[i] = im[j]; im[j] = ti;
                }
            }

            for (var len = 2; len <= n; len <<= 1)
            {
                var ang = -2f * Mathf.PI / len;
                var wlenR = Mathf.Cos(ang);
                var wlenI = Mathf.Sin(ang);
                for (var i = 0; i < n; i += len)
                {
                    var wR = 1f;
                    var wI = 0f;
                    for (var j = 0; j < len / 2; j++)
                    {
                        var uR = re[i + j];
                        var uI = im[i + j];
                        var vR = re[i + j + len / 2] * wR - im[i + j + len / 2] * wI;
                        var vI = re[i + j + len / 2] * wI + im[i + j + len / 2] * wR;
                        re[i + j] = uR + vR;
                        im[i + j] = uI + vI;
                        re[i + j + len / 2] = uR - vR;
                        im[i + j + len / 2] = uI - vI;

                        var nwR = wR * wlenR - wI * wlenI;
                        var nwI = wR * wlenI + wI * wlenR;
                        wR = nwR;
                        wI = nwI;
                    }
                }
            }
        }
    }
}
