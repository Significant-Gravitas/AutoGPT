"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

export interface WavyOrbSettings {
  opacity: number;
  haze: number;
  brightness: number;
  amplitude: number;
  spread: number;
  speed: number;
  edgeFade: number;
  tint: string;
  tintMix: number;
}

export const DEFAULT_WAVY_ORB_SETTINGS: WavyOrbSettings = {
  opacity: 1,
  haze: 0.25,
  brightness: 1,
  amplitude: 1,
  spread: 1,
  speed: 0.48,
  edgeFade: 0.82,
  tint: "#8b5cf6",
  tintMix: 0.83,
};

const VERTEX_SHADER = `#version 300 es
void main() {
  vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform float uTime;
uniform float uLow;
uniform float uMid;
uniform float uHigh;
uniform float uLevel;
uniform float uWake;
uniform float uWakeLag;
uniform float uOpacity;
uniform float uHaze;
uniform float uBrightness;
uniform float uAmplitude;
uniform float uSpread;
uniform float uSpeed;
uniform float uEdgeFade;
uniform vec3 uTint;
uniform float uTintMix;

out vec4 outColor;

const float PI = 3.14159265359;

vec3 spectral4(int sampleIndex) {
  vec3 c0 = vec3(0.26, 0.18, 1.0);
  vec3 c1 = vec3(0.74, 0.17, 0.96);
  vec3 c2 = vec3(1.0, 0.22, 0.52);
  vec3 c3 = vec3(1.0, 0.66, 0.22);
  return sampleIndex == 0 ? c0 : sampleIndex == 1 ? c1 : sampleIndex == 2 ? c2 : c3;
}

float waveY(float x, float amplitude, float envelope, float drift, float harmonic) {
  float fundamental = sin(x * 1.1 + drift);
  float partial = sin(x * 2.53 + drift * 1.6 + 1.7);
  float tilt = 1.0 + 0.14 * sin(x * 0.42 - drift * 0.6);
  return amplitude * envelope * tilt * (fundamental + harmonic * partial);
}

float thicknessAt(float xNormal, float mid) {
  float taper = 1.0 - 0.55 * clamp(abs(xNormal) * 0.75, 0.0, 1.0);
  return (0.012 + 0.009 * taper) * (1.0 + 0.2 * mid);
}

vec3 ribbon(
  vec2 p,
  float aspect,
  float amplitude,
  float spread,
  float drift,
  float harmonic,
  float mid,
  float level,
  float soften
) {
  float xNormal = p.x / max(aspect, 1.0);
  float envelope = cos(PI * 0.5 * min(abs(0.68 * xNormal), 1.0));
  envelope *= envelope;
  float thickness = thicknessAt(xNormal, mid) * soften;
  float hazeScale = mix(0.45, 2.4, uHaze);
  float soft = (0.009 + 0.006 * mid) * soften * hazeScale;
  float intensity = 0.019 * (1.0 + 0.45 * level);
  float mainWave = waveY(p.x, amplitude, envelope, drift, harmonic);
  vec3 numerator = vec3(0.0);
  vec3 denominator = vec3(0.0);

  for (int sampleIndex = 0; sampleIndex < 4; sampleIndex++) {
    vec3 hue = spectral4(sampleIndex);
    denominator += hue;
    float phase = mix(-spread, spread, float(sampleIndex) / 3.0);
    float shiftedWave = waveY(
      p.x,
      amplitude + 0.03 * mid,
      envelope,
      drift + phase,
      harmonic
    );
    float distanceToLine = abs(p.y - shiftedWave);
    float line = intensity /
      (sqrt(distanceToLine * distanceToLine + soft * soft) + thickness);
    line *= exp(-distanceToLine * distanceToLine);
    float low = min(mainWave, shiftedWave);
    float high = max(mainWave, shiftedWave);
    float outsideBand = max(0.0, max(p.y - high, low - p.y));
    float hazeFalloff = mix(0.026, 0.11, uHaze);
    float band = 4.2 * intensity * exp(-outsideBand / (hazeFalloff * soften));
    numerator += hue * (line + band);
  }

  float denominatorSum = (denominator.r + denominator.g + denominator.b) / 3.0;
  vec3 color = numerator / max(denominatorSum, 0.00001);
  float distanceToMain = abs(p.y - mainWave);
  color += 0.42 * intensity /
    (sqrt(distanceToMain * distanceToMain + soft * soft) + thickness);
  return color;
}

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  float aspect = uResolution.x / uResolution.y;
  vec2 p = (gl_FragCoord.xy + 0.5) * 2.0 / uResolution - 1.0;
  float screenX = abs(p.x);
  float screenY = p.y;
  p.x *= aspect;
  p /= 0.82;

  float wake = clamp(uWake, 0.0, 1.0);
  float amplitude = mix(0.06, 0.27 + 0.3 * uLow, wake) * uAmplitude;
  float lag = clamp(uWakeLag, 0.0, 1.0);
  float spread = mix(
    0.55,
    1.55 + 0.9 * uHigh + 0.35 * uMid + 0.7 * uLevel,
    lag
  ) * uSpread;
  float harmonic = mix(0.10, 0.22 + 0.12 * uHigh, wake);
  float xNormal = p.x / max(aspect, 1.0);
  float drift = uTime * uSpeed * mix(0.7, 1.35, wake);
  float ends = exp(-pow(xNormal * 1.05, 2.0));

  vec3 color = ribbon(
    p,
    aspect,
    amplitude,
    spread,
    drift,
    harmonic,
    uMid,
    uLevel,
    1.0
  );
  vec3 softRibbon = ribbon(
    p,
    aspect,
    amplitude * 0.94,
    spread * 1.05,
    drift - 0.18,
    harmonic * 0.8,
    uMid,
    uLevel,
    3.2 + 1.3 * uLevel
  );
  color += softRibbon * (mix(0.12, 0.34, uHaze) + 0.16 * uLevel);

  const float surface = 0.50;
  vec2 reflectedPoint = vec2(p.x, 2.0 * surface - p.y);
  vec3 reflection = ribbon(
    reflectedPoint,
    aspect,
    amplitude * 0.86,
    spread,
    drift,
    harmonic,
    uMid,
    uLevel,
    1.45
  );
  float underSurface = smoothstep(0.0, 0.16, p.y - surface);
  float depth = clamp((p.y - surface) / 0.95, 0.0, 1.0);
  color += reflection * 0.52 * underSurface * (1.0 - depth) * (1.0 - depth);
  color *= uBrightness;
  color = pow(max(color, 0.0), vec3(1.45));

  float above = smoothstep(1.0, 0.34, -screenY);
  float below = smoothstep(1.06, 0.52, screenY);
  float edge = screenY < 0.0 ? above : below;
  float sideFade = 1.0 - smoothstep(uEdgeFade, 1.0, screenX);
  color *= edge * ends * sideFade;

  float density = clamp(max(max(color.r, color.g), color.b) * 1.9, 0.0, 1.0);
  density = smoothstep(0.06, 0.92, density);
  vec3 hue = color / max(max(max(color.r, color.g), color.b), 0.000001);
  hue = mix(hue, uTint, uTintMix);
  vec3 paper = vec3(0.965, 0.969, 0.973);
  vec3 outputColor = mix(paper, paper * hue, 0.72);
  float grain = hash21(gl_FragCoord.xy * 0.75 + fract(uTime) * 91.7);
  outputColor += (grain - 0.5) / 255.0;
  outColor = vec4(clamp(outputColor, 0.0, 1.0), density * uOpacity);
}`;

interface Bands {
  low: number;
  mid: number;
  high: number;
  level: number;
}

interface Props {
  audioStream: MediaStream | null;
  settings: WavyOrbSettings;
}

function hexToRgb(hex: string) {
  const value = Number.parseInt(hex.slice(1), 16);
  return [
    ((value >> 16) & 255) / 255,
    ((value >> 8) & 255) / 255,
    (value & 255) / 255,
  ] as const;
}

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function follow(current: number, target: number, delta: number) {
  const rate = target > current ? 3.8 : 1.35;
  return current + (target - current) * Math.min(1, delta * rate);
}

function bandAverage(
  frequencies: Uint8Array<ArrayBuffer>,
  binHz: number,
  lowHz: number,
  highHz: number,
) {
  const start = Math.max(0, Math.floor(lowHz / binHz));
  const end = Math.min(frequencies.length, Math.ceil(highHz / binHz));
  if (end <= start) return 0;
  let sum = 0;
  for (let index = start; index < end; index++) {
    sum += frequencies[index];
  }
  return sum / (end - start) / 255;
}

export function WavyOrb({ audioStream, settings }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const settingsRef = useRef(settings);
  const prefersReducedMotion = useReducedMotion();
  const [isSupported, setIsSupported] = useState(true);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      premultipliedAlpha: false,
      powerPreference: "low-power",
    });
    if (!gl) {
      setIsSupported(false);
      return;
    }

    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragmentShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      FRAGMENT_SHADER,
    );
    if (!vertexShader || !fragmentShader) {
      setIsSupported(false);
      if (vertexShader) gl.deleteShader(vertexShader);
      if (fragmentShader) gl.deleteShader(fragmentShader);
      return;
    }

    const program = gl.createProgram();
    if (!program) {
      setIsSupported(false);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      return;
    }
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      setIsSupported(false);
      gl.deleteProgram(program);
      return;
    }

    const vertexArray = gl.createVertexArray();
    gl.bindVertexArray(vertexArray);
    gl.useProgram(program);
    const uniforms = {
      resolution: gl.getUniformLocation(program, "uResolution"),
      time: gl.getUniformLocation(program, "uTime"),
      low: gl.getUniformLocation(program, "uLow"),
      mid: gl.getUniformLocation(program, "uMid"),
      high: gl.getUniformLocation(program, "uHigh"),
      level: gl.getUniformLocation(program, "uLevel"),
      wake: gl.getUniformLocation(program, "uWake"),
      wakeLag: gl.getUniformLocation(program, "uWakeLag"),
      opacity: gl.getUniformLocation(program, "uOpacity"),
      haze: gl.getUniformLocation(program, "uHaze"),
      brightness: gl.getUniformLocation(program, "uBrightness"),
      amplitude: gl.getUniformLocation(program, "uAmplitude"),
      spread: gl.getUniformLocation(program, "uSpread"),
      speed: gl.getUniformLocation(program, "uSpeed"),
      edgeFade: gl.getUniformLocation(program, "uEdgeFade"),
      tint: gl.getUniformLocation(program, "uTint"),
      tintMix: gl.getUniformLocation(program, "uTintMix"),
    };
    const activeCanvas = canvas;
    const context = gl;

    let audioContext: AudioContext | null = null;
    let analyser: AnalyserNode | null = null;
    let frequencies: Uint8Array<ArrayBuffer> | null = null;
    if (audioStream && typeof AudioContext !== "undefined") {
      audioContext = new AudioContext();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.93;
      audioContext.createMediaStreamSource(audioStream).connect(analyser);
      frequencies = new Uint8Array(analyser.frequencyBinCount);
    }

    const bands: Bands = { low: 0, mid: 0, high: 0, level: 0 };
    let animationFrame = 0;
    let lastFrame = performance.now();
    let isVisible = true;
    let isIntersecting = true;
    let wake = 0;
    let wakeLag = 0;

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(activeCanvas.clientWidth * dpr));
      const height = Math.max(1, Math.round(activeCanvas.clientHeight * dpr));
      if (activeCanvas.width !== width || activeCanvas.height !== height) {
        activeCanvas.width = width;
        activeCanvas.height = height;
      }
      context.viewport(0, 0, width, height);
    }

    function readBands(now: number, delta: number) {
      if (analyser && frequencies && audioContext) {
        analyser.getByteFrequencyData(frequencies);
        const binHz = audioContext.sampleRate / analyser.fftSize;
        const low = bandAverage(frequencies, binHz, 60, 320);
        const mid = bandAverage(frequencies, binHz, 320, 1600);
        const high = bandAverage(frequencies, binHz, 1600, 6000);
        bands.low = follow(bands.low, low * 1.35, delta);
        bands.mid = follow(bands.mid, mid * 1.35, delta);
        bands.high = follow(bands.high, high * 1.45, delta);
        bands.level = follow(bands.level, (low + mid + high) * 0.58, delta);
        return;
      }

      const seconds = now / 1000;
      bands.low = 0.38 + Math.sin(seconds * 0.83) * 0.16;
      bands.mid = 0.32 + Math.sin(seconds * 1.37 + 1.2) * 0.13;
      bands.high = 0.26 + Math.sin(seconds * 2.11 + 2.4) * 0.11;
      bands.level = 0.38 + Math.sin(seconds * 0.61) * 0.12;
    }

    function draw(now: number) {
      resize();
      const delta = Math.min((now - lastFrame) / 1000, 0.1);
      lastFrame = now;
      readBands(now, delta);
      const wakeTarget = Math.min(1, bands.level * 1.35);
      const wakeRate = wakeTarget > wake ? 1.8 : 0.65;
      wake += (wakeTarget - wake) * Math.min(1, delta * wakeRate);
      wakeLag += (wake - wakeLag) * Math.min(1, delta * 2.8);
      context.uniform2f(
        uniforms.resolution,
        activeCanvas.width,
        activeCanvas.height,
      );
      context.uniform1f(uniforms.time, prefersReducedMotion ? 3.2 : now / 1000);
      context.uniform1f(uniforms.low, bands.low);
      context.uniform1f(uniforms.mid, bands.mid);
      context.uniform1f(uniforms.high, bands.high);
      context.uniform1f(uniforms.level, bands.level);
      context.uniform1f(uniforms.wake, wake);
      context.uniform1f(uniforms.wakeLag, wakeLag);
      const currentSettings = settingsRef.current;
      const tint = hexToRgb(currentSettings.tint);
      context.uniform1f(uniforms.opacity, currentSettings.opacity);
      context.uniform1f(uniforms.haze, currentSettings.haze);
      context.uniform1f(uniforms.brightness, currentSettings.brightness);
      context.uniform1f(uniforms.amplitude, currentSettings.amplitude);
      context.uniform1f(uniforms.spread, currentSettings.spread);
      context.uniform1f(uniforms.speed, currentSettings.speed);
      context.uniform1f(uniforms.edgeFade, currentSettings.edgeFade);
      context.uniform3f(uniforms.tint, tint[0], tint[1], tint[2]);
      context.uniform1f(uniforms.tintMix, currentSettings.tintMix);
      context.drawArrays(context.TRIANGLES, 0, 3);
    }

    function render(now: number) {
      if (!isVisible || !isIntersecting) return;
      draw(now);
      if (!prefersReducedMotion) {
        animationFrame = requestAnimationFrame(render);
      }
    }

    function syncAnimation() {
      isVisible = document.visibilityState !== "hidden";
      cancelAnimationFrame(animationFrame);
      if (isVisible && isIntersecting) {
        animationFrame = requestAnimationFrame(render);
      }
    }

    const resizeObserver = new ResizeObserver(() => draw(performance.now()));
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      isIntersecting = entry.isIntersecting;
      syncAnimation();
    });
    resizeObserver.observe(activeCanvas);
    intersectionObserver.observe(activeCanvas);
    document.addEventListener("visibilitychange", syncAnimation);
    draw(performance.now());
    syncAnimation();

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      document.removeEventListener("visibilitychange", syncAnimation);
      void audioContext?.close();
      if (vertexArray) context.deleteVertexArray(vertexArray);
      context.deleteProgram(program);
    };
  }, [audioStream, prefersReducedMotion]);

  if (!isSupported) {
    return (
      <div
        data-testid="orb-wavy"
        className="relative h-full w-full overflow-hidden"
        aria-hidden
      >
        <div className="absolute left-0 top-1/2 h-1 w-full -translate-y-1/2 bg-gradient-to-r from-transparent via-purple-500 to-transparent blur-sm" />
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      data-testid="orb-wavy"
      className="block h-full w-full bg-transparent"
      aria-hidden
    />
  );
}
