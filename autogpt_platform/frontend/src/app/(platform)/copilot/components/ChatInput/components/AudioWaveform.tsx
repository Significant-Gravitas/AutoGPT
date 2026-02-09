"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  stream: MediaStream | null;
  barCount?: number;
  barWidth?: number;
  barGap?: number;
  barColor?: string;
  minBarHeight?: number;
  maxBarHeight?: number;
}

export function AudioWaveform({
  stream,
  barCount = 24,
  barWidth = 3,
  barGap = 2,
  barColor = "#ef4444", // red-500
  minBarHeight = 4,
  maxBarHeight = 32,
}: Props) {
  const [bars, setBars] = useState<number[]>(() =>
    Array(barCount).fill(minBarHeight),
  );
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    if (!stream) {
      setBars(Array(barCount).fill(minBarHeight));
      return;
    }

    // Create audio context and analyser
    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.3;

    // Connect the stream to the analyser
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    audioContextRef.current = audioContext;
    analyserRef.current = analyser;
    sourceRef.current = source;

    const timeData = new Uint8Array(analyser.frequencyBinCount);

    const updateBars = () => {
      if (!analyserRef.current) return;

      analyserRef.current.getByteTimeDomainData(timeData);

      // Distribute time-domain data across bars
      // This shows waveform amplitude, making all bars respond to audio
      const newBars: number[] = [];
      const samplesPerBar = timeData.length / barCount;

      for (let i = 0; i < barCount; i++) {
        // Sample waveform data for this bar
        let maxAmplitude = 0;
        const startIdx = Math.floor(i * samplesPerBar);
        const endIdx = Math.floor((i + 1) * samplesPerBar);

        for (let j = startIdx; j < endIdx && j < timeData.length; j++) {
          // Convert to amplitude (distance from center 128)
          const amplitude = Math.abs(timeData[j] - 128);
          maxAmplitude = Math.max(maxAmplitude, amplitude);
        }

        // Normalize amplitude (0-128 range) to 0-1
        const normalized = maxAmplitude / 128;
        // Apply sensitivity boost (multiply by 4) and use sqrt curve to amplify quiet sounds
        const boosted = Math.min(1, Math.sqrt(normalized) * 4);
        const height = minBarHeight + boosted * (maxBarHeight - minBarHeight);
        newBars.push(height);
      }

      setBars(newBars);
      animationRef.current = requestAnimationFrame(updateBars);
    };

    updateBars();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      analyserRef.current = null;
      audioContextRef.current = null;
      sourceRef.current = null;
    };
  }, [stream, barCount, minBarHeight, maxBarHeight]);

  const totalWidth = barCount * barWidth + (barCount - 1) * barGap;

  return (
    <div
      className="flex items-center justify-center"
      style={{
        width: totalWidth,
        height: maxBarHeight,
        gap: barGap,
      }}
    >
      {bars.map((height, i) => {
        const barHeight = Math.max(minBarHeight, height);
        return (
          <div
            key={i}
            className="relative"
            style={{
              width: barWidth,
              height: maxBarHeight,
            }}
          >
            <div
              className="absolute left-0 rounded-full transition-[height] duration-75"
              style={{
                width: barWidth,
                height: barHeight,
                top: "50%",
                transform: "translateY(-50%)",
                backgroundColor: barColor,
              }}
            />
          </div>
        );
      })}
    </div>
  );
}
