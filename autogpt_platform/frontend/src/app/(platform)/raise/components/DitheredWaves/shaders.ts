export const VERTEX_SHADER = `#version 300 es
const vec2 verts[3] = vec2[](
  vec2(-1.0, -1.0),
  vec2( 3.0, -1.0),
  vec2(-1.0,  3.0)
);
void main() {
  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}`;

export const FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform float u_time;
uniform vec3 uColor0;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec3 uPrevColor0;
uniform vec3 uPrevColor1;
uniform vec3 uPrevColor2;
uniform vec3 uPrevColor3;
// Position of the wipe front along the bottom-left → top-right diagonal.
uniform float uWipe;

out vec4 fragColor;

vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec2 fade(vec2 t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float cnoise(vec2 P) {
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi);
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;
  vec4 i = permute(permute(ix) + iy);
  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;
  vec2 g00 = vec2(gx.x, gy.x);
  vec2 g10 = vec2(gx.y, gy.y);
  vec2 g01 = vec2(gx.z, gy.z);
  vec2 g11 = vec2(gx.w, gy.w);
  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;
  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));
  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

const int OCTAVES = 8;

float fbm(vec2 p) {
  float value = -0.2;
  float amplitude = 1.0;
  float frequency = 2.5;
  for (int i = 0; i < OCTAVES; i++) {
    value += amplitude * abs(cnoise(p));
    p *= frequency;
    amplitude *= 0.35;
  }
  return value;
}

float pattern(vec2 p) {
  vec2 p2 = p - u_time * 0.012;
  return fbm(p - fbm(p + fbm(p2)));
}

float bayer8x8(vec2 fragCoord) {
  const float m[64] = float[64](
     0.0/64.0, 32.0/64.0,  8.0/64.0, 40.0/64.0,  2.0/64.0, 34.0/64.0, 10.0/64.0, 42.0/64.0,
    48.0/64.0, 16.0/64.0, 56.0/64.0, 24.0/64.0, 50.0/64.0, 18.0/64.0, 58.0/64.0, 26.0/64.0,
    12.0/64.0, 44.0/64.0,  4.0/64.0, 36.0/64.0, 14.0/64.0, 46.0/64.0,  6.0/64.0, 38.0/64.0,
    60.0/64.0, 28.0/64.0, 52.0/64.0, 20.0/64.0, 62.0/64.0, 30.0/64.0, 54.0/64.0, 22.0/64.0,
     3.0/64.0, 35.0/64.0, 11.0/64.0, 43.0/64.0,  1.0/64.0, 33.0/64.0,  9.0/64.0, 41.0/64.0,
    51.0/64.0, 19.0/64.0, 59.0/64.0, 27.0/64.0, 49.0/64.0, 17.0/64.0, 57.0/64.0, 25.0/64.0,
    15.0/64.0, 47.0/64.0,  7.0/64.0, 39.0/64.0, 13.0/64.0, 45.0/64.0,  5.0/64.0, 37.0/64.0,
    63.0/64.0, 31.0/64.0, 55.0/64.0, 23.0/64.0, 61.0/64.0, 29.0/64.0, 53.0/64.0, 21.0/64.0
  );
  int x = int(fragCoord.x) % 8;
  int y = int(fragCoord.y) % 8;
  return m[y * 8 + x];
}

vec3 ramp(float t, vec3 c0, vec3 c1, vec3 c2, vec3 c3) {
  float s = t * 3.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  return mix(c2, c3, s - 2.0);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  uv -= 0.5;
  uv.x *= uResolution.x / uResolution.y;

  float f = pattern(uv);
  float lum = clamp(f, 0.0, 1.0);

  float threshold = bayer8x8(gl_FragCoord.xy) - 0.75;
  float v = clamp(lum + threshold, 0.0, 1.0);
  v = floor(v * 3.0 + 0.5) / 3.0;

  // 0 at the bottom-left corner, 1 at the top-right, so the wipe front
  // sweeps the whole surface on one diagonal pass.
  vec2 st = gl_FragCoord.xy / uResolution.xy;
  float diagonal = (st.x + st.y) * 0.5;
  // Edges must stay ordered: smoothstep is undefined for edge0 >= edge1, so the
  // descending ramp is expressed as an inverted ascending one.
  float front = 1.0 - smoothstep(uWipe - 0.12, uWipe + 0.12, diagonal);

  vec3 previous = ramp(v, uPrevColor0, uPrevColor1, uPrevColor2, uPrevColor3);
  vec3 next = ramp(v, uColor0, uColor1, uColor2, uColor3);

  fragColor = vec4(mix(previous, next, front), 1.0);
}`;
