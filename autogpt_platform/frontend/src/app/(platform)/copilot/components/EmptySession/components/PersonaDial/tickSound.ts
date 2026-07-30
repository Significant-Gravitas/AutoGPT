// Synthesised rotary-dial tick — a short, quiet high-frequency blip played
// each time the wheel crosses a slot. No audio asset needed.

let audioContext: AudioContext | null = null;
let lastTickAt = 0;

export function playDialTick() {
  if (typeof window === "undefined") return;
  const now = performance.now();
  // Cap the rate so a fast spin sounds like a ratchet, not a buzz.
  if (now - lastTickAt < 30) return;
  lastTickAt = now;

  try {
    audioContext ??= new AudioContext();
    if (audioContext.state === "suspended") void audioContext.resume();

    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    osc.type = "triangle";
    osc.frequency.value = 1900;
    gain.gain.setValueAtTime(0.06, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(
      0.0001,
      audioContext.currentTime + 0.045,
    );
    osc.connect(gain);
    gain.connect(audioContext.destination);
    osc.start();
    osc.stop(audioContext.currentTime + 0.05);
  } catch {
    // No audio output / autoplay policy blocked it — the dial works silently.
  }
}
