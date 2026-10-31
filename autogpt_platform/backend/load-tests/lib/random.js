// Deterministic choices, so two runs with the same RUN_SEED browse the same
// skills, open the same agents and send the same messages in the same order.
// k6's Math.random cannot be seeded, so every choice in the scenarios goes
// through a generator seeded from RUN_SEED, the scenario and the iteration.

const DEFAULT_SEED = "autogpt-load-tests";

export function runSeed() {
  return __ENV.RUN_SEED || DEFAULT_SEED;
}

// FNV-1a, 32-bit.
export function hashString(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

// mulberry32: a small, fast generator with a 32-bit state.
export function seededRandom(seed) {
  let state =
    (typeof seed === "number" ? seed : hashString(String(seed))) >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The generator for one iteration of one scenario.
export function iterationRandom(scenarioName, iteration) {
  return seededRandom(`${runSeed()}:${scenarioName}:${iteration}`);
}

export function pick(random, items) {
  if (!items || items.length === 0) return undefined;
  return items[Math.floor(random() * items.length) % items.length];
}

export function pickIndex(random, length) {
  return Math.floor(random() * length) % length;
}
