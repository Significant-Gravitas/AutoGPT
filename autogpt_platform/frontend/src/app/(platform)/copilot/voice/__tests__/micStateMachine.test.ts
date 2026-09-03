import { describe, expect, it } from "vitest";

import {
  isMicOpen,
  voiceReduce,
  type VoiceEvent,
  type VoiceState,
} from "../micStateMachine";

function run(events: VoiceEvent["type"][], from: VoiceState = "off") {
  return events.reduce<VoiceState>(
    (state, type) => voiceReduce(state, { type } as VoiceEvent),
    from,
  );
}

describe("voiceReduce", () => {
  it("walks a full turn back to listening", () => {
    expect(
      run([
        "ENABLE",
        "SPEECH_START",
        "SPEECH_END",
        "TRANSCRIPT_SENT",
        "REPLY_SPEAKING",
        "REPLY_DONE",
      ]),
    ).toBe("listening");
  });

  it("ignores everything but ENABLE while off", () => {
    expect(run(["SPEECH_START"])).toBe("off");
    expect(run(["REPLY_SPEAKING"])).toBe("off");
  });

  it("returns to listening when the VAD misfires", () => {
    expect(run(["ENABLE", "SPEECH_START", "SPEECH_MISFIRE"])).toBe("listening");
  });

  it("returns to listening when the transcript is dropped", () => {
    expect(
      run(["ENABLE", "SPEECH_START", "SPEECH_END", "TRANSCRIPT_DROPPED"]),
    ).toBe("listening");
  });

  it("reopens the mic when a reply produced no speech at all", () => {
    expect(
      run([
        "ENABLE",
        "SPEECH_START",
        "SPEECH_END",
        "TRANSCRIPT_SENT",
        "REPLY_DONE",
      ]),
    ).toBe("listening");
  });

  it("stays speaking across several chunks", () => {
    expect(
      run([
        "ENABLE",
        "SPEECH_START",
        "SPEECH_END",
        "TRANSCRIPT_SENT",
        "REPLY_SPEAKING",
        "REPLY_SPEAKING",
      ]),
    ).toBe("speaking");
  });

  it("interrupts straight back to listening from any state", () => {
    expect(run(["ENABLE", "SPEECH_START", "SPEECH_END", "INTERRUPT"])).toBe(
      "listening",
    );
  });

  it("turns off from anywhere", () => {
    expect(run(["ENABLE", "SPEECH_START", "SPEECH_END", "DISABLE"])).toBe(
      "off",
    );
  });

  it("cannot skip states out of order", () => {
    expect(run(["ENABLE", "SPEECH_END"])).toBe("listening");
    expect(run(["ENABLE", "TRANSCRIPT_SENT"])).toBe("listening");
    expect(run(["ENABLE", "REPLY_SPEAKING"])).toBe("listening");
  });
});

describe("mic gating", () => {
  it("keeps the mic shut for every state that follows the user speaking", () => {
    expect(isMicOpen("listening")).toBe(true);
    expect(isMicOpen("hearing")).toBe(true);
    for (const state of [
      "off",
      "transcribing",
      "thinking",
      "speaking",
    ] as const) {
      expect(isMicOpen(state)).toBe(false);
    }
  });
});
