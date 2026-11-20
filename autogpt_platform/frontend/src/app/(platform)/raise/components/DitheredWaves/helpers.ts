import * as Sentry from "@sentry/nextjs";
import { FRAGMENT_SHADER, VERTEX_SHADER } from "./shaders";

export const DITHER_COLORS = [
  "#a78bfa",
  "#c4b5fd",
  "#ede9fe",
  "#ffffff",
] as const;

export function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.replace("#", ""), 16);
  return [
    ((value >> 16) & 255) / 255,
    ((value >> 8) & 255) / 255,
    (value & 255) / 255,
  ];
}

export function createDitherProgram(gl: WebGL2RenderingContext) {
  const vert = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  const program = gl.createProgram();

  if (!vert || !frag || !program) {
    if (vert) gl.deleteShader(vert);
    if (frag) gl.deleteShader(frag);
    if (program) gl.deleteProgram(program);
    return null;
  }

  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);

  // The program is never re-linked, so the linked binary is all we need to keep.
  gl.detachShader(program, vert);
  gl.detachShader(program, frag);
  gl.deleteShader(vert);
  gl.deleteShader(frag);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    reportShaderFailure("link", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
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
    reportShaderFailure("compile", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

function reportShaderFailure(stage: string, infoLog: string | null) {
  Sentry.captureException(
    new Error(
      `DitheredWaves shader ${stage} failed: ${infoLog || "no info log available"}`,
    ),
  );
}
