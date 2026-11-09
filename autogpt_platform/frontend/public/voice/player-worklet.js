// Playback for the onboarding voice interview: 24 kHz PCM16 chunks from the
// relay into a ring buffer, read out at the context's rate with linear
// interpolation. "flush" drops everything queued, which is what barge-in needs
// when the relay reports an interruption.

// The ring holds 30 s of Otto's audio (720,000 samples, about 2.9 MB at
// 24 kHz): the relay's re-attach replay, up to 20 s, and 10 s more arriving
// ahead of playback. Past that, the oldest audio waiting is dropped, so
// playback keeps to the newest rather than reading over what was written.
const RING_SECONDS = 30;

class PcmPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.sourceRate =
      (options.processorOptions && options.processorOptions.sourceRate) ||
      24000;
    this.step = this.sourceRate / sampleRate;
    this.size = this.sourceRate * RING_SECONDS;
    this.ring = new Float32Array(this.size);
    this.written = 0;
    this.read = 0;
    this.sinceReport = 0;
    this.port.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const pcm = new Int16Array(event.data);
        for (let i = 0; i < pcm.length; i++) {
          this.ring[this.written % this.size] = pcm[i] / 32768;
          this.written += 1;
        }
        // One sample short of the ring, so the interpolation's next sample
        // is never one written over.
        if (this.written - this.read > this.size - 1) {
          this.read = this.written - (this.size - 1);
        }
      } else if (event.data && event.data.type === "flush") {
        this.read = this.written;
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    for (let k = 0; k < out.length; k++) {
      if (this.read + 1 < this.written) {
        const i = Math.floor(this.read);
        const frac = this.read - i;
        out[k] =
          this.ring[i % this.size] * (1 - frac) +
          this.ring[(i + 1) % this.size] * frac;
        this.read += this.step;
      } else {
        out[k] = 0;
      }
    }
    this.sinceReport += out.length;
    if (this.sinceReport >= sampleRate / 4) {
      this.sinceReport = 0;
      this.port.postMessage({
        playedS: this.read / this.sourceRate,
        bufferedS: Math.max(0, this.written - this.read) / this.sourceRate,
      });
    }
    return true;
  }
}

registerProcessor("pcm-player", PcmPlayer);
