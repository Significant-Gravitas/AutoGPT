// Microphone capture for the onboarding voice interview: the context's sample
// rate (usually 48 kHz) down to 16 kHz PCM16, posted to the page in 40 ms
// frames. Each output sample is the mean of the input samples it covers, a box
// filter that keeps aliasing low enough for speech and works for non-integer
// ratios such as 44.1 kHz.
class MicCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    const target = opts.targetRate || 16000;
    this.ratio = sampleRate / target;
    this.frame = new Int16Array(
      Math.round((target * (opts.frameMs || 40)) / 1000),
    );
    this.filled = 0;
    this.pos = 0;
    this.acc = 0;
    this.count = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.acc += channel[i];
      this.count += 1;
      this.pos += 1;
      if (this.pos < this.ratio) continue;
      this.pos -= this.ratio;
      const v = Math.max(-1, Math.min(1, this.acc / this.count));
      this.acc = 0;
      this.count = 0;
      this.frame[this.filled++] = v < 0 ? v * 0x8000 : v * 0x7fff;
      if (this.filled === this.frame.length) {
        const out = this.frame.slice();
        this.port.postMessage(out.buffer, [out.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
