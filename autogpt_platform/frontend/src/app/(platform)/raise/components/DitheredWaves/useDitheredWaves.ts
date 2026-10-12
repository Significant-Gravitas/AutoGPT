import { useEffect, useRef } from "react";
import { createDitherProgram, DITHER_COLORS, hexToRgb } from "./helpers";

const WIPE_DURATION_MS = 1100;
// Matches the smoothstep width in the shader, so the front starts fully off
// one corner and ends fully off the other.
const WIPE_EDGE = 0.12;

export function useDitheredWaves(colors: readonly string[]) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const colorsKey = colors.join(",");
  const colorsRef = useRef(colors);
  const previousColorsRef = useRef(colors);
  const wipeStartRef = useRef<number | null>(null);

  useEffect(() => {
    if (colorsKey === colorsRef.current.join(",")) return;
    previousColorsRef.current = colorsRef.current;
    colorsRef.current = colors;
    wipeStartRef.current = performance.now();
  }, [colors, colorsKey]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas?.getContext("webgl2", { antialias: false });
    if (!canvas || !gl) return;

    const program = createDitherProgram(gl);
    if (!program) return;

    gl.useProgram(program);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const resolutionLocation = gl.getUniformLocation(program, "uResolution");
    const timeLocation = gl.getUniformLocation(program, "u_time");
    const wipeLocation = gl.getUniformLocation(program, "uWipe");
    const colorLocations = DITHER_COLORS.map((_, index) =>
      gl.getUniformLocation(program, `uColor${index}`),
    );
    const previousColorLocations = DITHER_COLORS.map((_, index) =>
      gl.getUniformLocation(program, `uPrevColor${index}`),
    );

    let width = 0;
    let height = 0;

    function resize() {
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, Math.floor(rect.width * pixelRatio));
      height = Math.max(1, Math.floor(rect.height * pixelRatio));
      canvas.width = width;
      canvas.height = height;
    }

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const start = performance.now();
    let frame = 0;

    function paletteAt(
      locations: (WebGLUniformLocation | null)[],
      palette: readonly string[],
    ) {
      if (!gl) return;
      locations.forEach((location, index) => {
        const [r, g, b] = hexToRgb(palette[index] ?? DITHER_COLORS[index]);
        gl.uniform3f(location, r, g, b);
      });
    }

    function render() {
      if (!gl) return;
      const now = performance.now();
      const wipeStart = wipeStartRef.current;
      const progress = wipeStart
        ? Math.min(1, (now - wipeStart) / WIPE_DURATION_MS)
        : 1;

      gl.viewport(0, 0, width, height);
      gl.uniform2f(resolutionLocation, width, height);
      gl.uniform1f(timeLocation, (now - start) / 1000);
      // Map 0..1 onto a range that clears the smoothstep band at both ends.
      gl.uniform1f(wipeLocation, progress * (1 + 2 * WIPE_EDGE) - WIPE_EDGE);
      paletteAt(colorLocations, colorsRef.current);
      paletteAt(previousColorLocations, previousColorsRef.current);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      frame = requestAnimationFrame(render);
    }

    render();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      gl.deleteProgram(program);
      gl.deleteVertexArray(vao);
    };
  }, []);

  return canvasRef;
}
