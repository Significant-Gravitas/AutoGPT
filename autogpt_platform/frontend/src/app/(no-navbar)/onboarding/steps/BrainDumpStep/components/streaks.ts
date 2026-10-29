// Geometry and colour for the voice streaks: thin gradient comets that orbit
// an avatar in three dimensions while the user speaks. Each orbit is a
// tilted circle around the body's centre; the comet passes behind the
// avatar on the far side and in front on the near side.

export const MAX_STREAKS = 10;

// Gap between the avatar's edge and the orbit, as a fraction of its radius.
const CLEARANCE_MIN = 0.15;
const CLEARANCE_MAX = 0.22;
// Directions sampled around the body. The avatar is a disc, so every sample
// is the same radius — the array is kept so the orbit maths stays general.
const PROFILE_STEPS = 72;
// Projected ellipse is about 40% as wide as it is long.
const ORBIT_TILT = 0.42;
// How strongly depth changes apparent size: the near side reads about 25%
// larger than the far side.
const PERSPECTIVE = 0.25;
// Orbit angle at which the comet is furthest behind the head.
const BEHIND = (3 * Math.PI) / 2;

// Everything about the space the comets fly in: centred on the avatar, just
// outside its disc, wearing whatever two tones it was given.
export interface StreakField {
  size: number;
  pad: number;
  box: number;
  centre: { x: number; y: number };
  // Distance from the centre to the avatar's edge, in px, for PROFILE_STEPS
  // evenly spaced directions starting at 3 o'clock and going clockwise.
  profile: number[];
  clearanceMin: number;
  clearanceMax: number;
  colors: [string, string];
}

export function createStreakField(
  colors: [string, string],
  size: number,
): StreakField {
  const radius = size / 2;
  const pad = size / 4;
  return {
    size,
    pad,
    box: size + pad * 2,
    centre: { x: radius, y: radius },
    profile: Array.from({ length: PROFILE_STEPS }, () => radius),
    clearanceMin: CLEARANCE_MIN * radius,
    clearanceMax: CLEARANCE_MAX * radius,
    colors,
  };
}

export function bodyRadiusAt(field: StreakField, angle: number) {
  const steps = field.profile.length;
  const turns = (((angle / (Math.PI * 2)) % 1) + 1) % 1;
  const position = turns * steps;
  const before = Math.floor(position) % steps;
  const after = (before + 1) % steps;
  const blend = position - Math.floor(position);
  return field.profile[before] * (1 - blend) + field.profile[after] * blend;
}

export interface Streak {
  id: number;
  // Half the ellipse's long axis, in px.
  axis: number;
  // Where the ellipse is centred, relative to the body's centre, in px: it
  // slides along its long axis so both ends clear the body by the same gap.
  shift: { x: number; y: number };
  // Tilt of the ring out of the horizontal (x/z) plane, in radians.
  tilt: number;
  // Roll of the ring about the viewing axis, in radians.
  roll: number;
  start: number;
  sweep: number;
  bornAt: number;
  durationMs: number;
  colors: [string, string];
  width: number;
  tailAngle: number;
}

export interface Point3 {
  x: number;
  y: number;
  z: number;
}

function rand(min: number, max: number, random: () => number) {
  return min + (max - min) * random();
}

// Comets always come as a mirrored pair, one on each ellipse, born in the
// same frame with the same start, speed, length and lifetime, so the two
// orbits light up together rather than at random.
export function makeStreakPair(
  field: StreakField,
  firstId: number,
  loudness: number,
  now: number,
  random: () => number = Math.random,
): [Streak, Streak] {
  const clearance = rand(field.clearanceMin, field.clearanceMax, random);
  // Born at the deepest point behind the head, and one full lap brings the
  // comet back there, so it appears and disappears out of sight.
  const start = BEHIND + rand(-0.25, 0.25, random);
  const sweep = Math.PI * 2 * (random() < 0.5 ? -1 : 1);
  const tilt = ORBIT_TILT * (random() < 0.5 ? -1 : 1);
  const widthScale = field.size / 160;
  const shared = {
    start,
    bornAt: now,
    durationMs: rand(1500, 2100, random),
    width: rand(3, 5, random) * widthScale,
    tailAngle: tailFor(loudness),
    colors: field.colors,
  };
  const forward = fitEllipse(field, Math.PI / 4, clearance);
  const backward = fitEllipse(field, -Math.PI / 4, clearance);
  return [
    { ...shared, ...forward, id: firstId, roll: Math.PI / 4, tilt, sweep },
    {
      ...shared,
      ...backward,
      id: firstId + 1,
      roll: -Math.PI / 4,
      tilt: -tilt,
      sweep: -sweep,
    },
  ];
}

// Sizes one smooth ellipse whose long axis lies along `roll`: it reaches
// `clearance` past the body at both ends, sliding its centre along the axis
// when the body reaches further one way (a dome's square base) than the
// other (its round top).
function fitEllipse(field: StreakField, roll: number, clearance: number) {
  const ahead = bodyRadiusAt(field, roll) + clearance;
  const behind = bodyRadiusAt(field, roll + Math.PI) + clearance;
  const offset = (ahead - behind) / 2;
  return {
    axis: (ahead + behind) / 2,
    shift: { x: Math.cos(roll) * offset, y: Math.sin(roll) * offset },
  };
}

// A point on the unit orbit at angle `theta`, z toward the viewer. The
// orbit's real extent is applied in `project`, where it follows the body.
export function orbitPoint(streak: Streak, theta: number): Point3 {
  // Horizontal ring around the head, then tilted, then rolled.
  const x0 = Math.cos(theta);
  const z0 = Math.sin(theta);
  const y1 = -z0 * Math.sin(streak.tilt);
  const z1 = z0 * Math.cos(streak.tilt);
  const x2 = x0 * Math.cos(streak.roll) - y1 * Math.sin(streak.roll);
  const y2 = x0 * Math.sin(streak.roll) + y1 * Math.cos(streak.roll);
  return { x: x2, y: y2, z: z1 };
}

// Places a unit-orbit point on screen: scaled to the comet's ellipse, moved
// to the ellipse's centre, and pushed out or pulled in by depth.
export function project(field: StreakField, streak: Streak, point: Point3) {
  const scale = 1 + point.z * PERSPECTIVE;
  return {
    x: field.centre.x + streak.shift.x + point.x * streak.axis * scale,
    y: field.centre.y + streak.shift.y + point.y * streak.axis * scale,
    scale,
  };
}

function easeInOut(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Where the comet's head is, given how far through its life it is. The lap
// starts slowly out of sight, speeds through the front and eases back in.
export function headAngle(streak: Streak, progress: number) {
  return streak.start + streak.sweep * easeInOut(progress);
}

// How much of the tail is drawn: it grows out of the head at the start and
// is drawn back into the head at the end, both while still behind the body,
// so the comet never cuts off in front.
export function tailSpan(streak: Streak, progress: number) {
  if (progress <= 0 || progress >= 1) return 0;
  if (progress < 0.18) return streak.tailAngle * (progress / 0.18);
  if (progress > 0.8) return streak.tailAngle * ((1 - progress) / 0.2);
  return streak.tailAngle;
}

// Overall visibility: a short fade at either end, both spent behind the head.
export function envelope(progress: number) {
  if (progress <= 0 || progress >= 1) return 0;
  if (progress < 0.08) return progress / 0.08;
  if (progress > 0.92) return (1 - progress) / 0.08;
  return 1;
}

const SILENCE = 0.06;
// In silence a comet is just a dot; a voice stretches it into a wave.
const DOT_TAIL = 0.06;

export function tailFor(loudness: number) {
  if (loudness < SILENCE) return DOT_TAIL;
  return 1.4 + 0.8 * Math.min(1, loudness);
}

// Streaks per second for a given loudness: a slow trickle of dots in
// silence, a few waves for a murmur, a shower for a raised voice.
export function spawnRate(loudness: number) {
  if (loudness < SILENCE) return 0.7;
  return 2 + 12 * Math.min(1, loudness);
}

export function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export interface StreakStore {
  streaks: Streak[];
  nextId: number;
  budget: number;
  lastTick: number;
}

export function createStreakStore(): StreakStore {
  return { streaks: [], nextId: 0, budget: 0, lastTick: 0 };
}

export function advanceStreaks(
  store: StreakStore,
  field: StreakField,
  loudness: number,
  now: number,
) {
  const seconds = store.lastTick
    ? Math.min(0.1, Math.max(0, (now - store.lastTick) / 1000))
    : 0;
  store.lastTick = now;
  store.streaks = store.streaks.filter(
    (streak) => now - streak.bornAt < streak.durationMs,
  );
  store.budget = Math.min(
    MAX_STREAKS / 2,
    store.budget + seconds * spawnRate(loudness),
  );
  const pairs = Math.min(
    Math.floor(store.budget),
    Math.floor((MAX_STREAKS - store.streaks.length) / 2),
  );
  store.budget %= 1;
  for (let index = 0; index < pairs; index++) {
    store.streaks.push(...makeStreakPair(field, store.nextId, loudness, now));
    store.nextId += 2;
  }
}
