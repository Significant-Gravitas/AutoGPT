import { describe, expect, it } from "vitest";
import {
  addOttoStats,
  newOttoListening,
  ottoListeningSummary,
} from "../ottoStats";

function report(stats: Record<string, unknown>[]) {
  return new Map(
    stats.map((stat) => [String(stat.id), stat]),
  ) as unknown as RTCStatsReport;
}

function ottoAudio(id: string, counts: Record<string, unknown>) {
  return { id, type: "inbound-rtp", kind: "audio", ...counts };
}

describe("how Otto's speech fared in the call's room", () => {
  it("reads the share made up for lost packets, the packets lost, and the worst jitter and round trip", () => {
    const listening = newOttoListening();

    addOttoStats(
      listening,
      report([
        ottoAudio("IT01A1", {
          concealedSamples: 480,
          totalSamplesReceived: 48_000,
          packetsLost: 3,
          jitter: 0.012,
        }),
        { id: "T01", type: "transport", selectedCandidatePairId: "CP01" },
        { id: "CP01", type: "candidate-pair", currentRoundTripTime: 0.08 },
        // A pair the transport does not use says nothing of the call.
        { id: "CP02", type: "candidate-pair", currentRoundTripTime: 0.9 },
      ]),
    );
    addOttoStats(
      listening,
      report([
        ottoAudio("IT01A1", {
          concealedSamples: 4_800,
          totalSamplesReceived: 96_000,
          packetsLost: 12,
          jitter: 0.004,
        }),
        { id: "T01", type: "transport", selectedCandidatePairId: "CP01" },
        { id: "CP01", type: "candidate-pair", currentRoundTripTime: 0.142 },
      ]),
    );

    // A stream's counts run on, so its latest stand; the worst jitter and
    // round trip read stand.
    expect(ottoListeningSummary(listening)).toEqual({
      concealed_share: 0.05,
      packets_lost: 12,
      jitter_ms_max: 12,
      rtt_ms_max: 142,
    });
  });

  it("sums the streams of every track Otto spoke on", () => {
    const listening = newOttoListening();

    addOttoStats(
      listening,
      report([
        ottoAudio("IT01A1", {
          concealedSamples: 1_000,
          totalSamplesReceived: 10_000,
          packetsLost: 4,
        }),
      ]),
    );
    // A rejoin, or LiveKit reconnecting, brings a track that counts from
    // nothing.
    addOttoStats(
      listening,
      report([
        ottoAudio("IT01A2", {
          concealedSamples: 0,
          totalSamplesReceived: 30_000,
          packetsLost: 1,
        }),
      ]),
    );

    expect(ottoListeningSummary(listening)).toMatchObject({
      concealed_share: 0.025,
      packets_lost: 5,
    });
  });

  it("reads the pair Firefox marks selected when no transport names one", () => {
    const listening = newOttoListening();

    addOttoStats(
      listening,
      report([
        { id: "CP01", type: "candidate-pair", currentRoundTripTime: 0.5 },
        {
          id: "CP02",
          type: "candidate-pair",
          selected: true,
          currentRoundTripTime: 0.06,
        },
      ]),
    );

    expect(ottoListeningSummary(listening).rtt_ms_max).toBe(60);
  });

  it("leaves out what is not Otto's audio, and counts that cannot be read", () => {
    const listening = newOttoListening();

    addOttoStats(
      listening,
      report([
        { id: "IT02V1", type: "inbound-rtp", kind: "video", packetsLost: 99 },
        { id: "OT01A1", type: "outbound-rtp", kind: "audio", packetsLost: 99 },
        ottoAudio("IT01A1", {
          concealedSamples: "many",
          totalSamplesReceived: 1_000,
          packetsLost: -2,
          jitter: Number.NaN,
        }),
      ]),
    );

    expect(ottoListeningSummary(listening)).toEqual({
      concealed_share: 0,
      packets_lost: 0,
      jitter_ms_max: null,
      rtt_ms_max: null,
    });
  });

  it("says nothing of a call whose stats were never read", () => {
    const empty = {
      concealed_share: null,
      packets_lost: null,
      jitter_ms_max: null,
      rtt_ms_max: null,
    };

    expect(ottoListeningSummary(newOttoListening())).toEqual(empty);
    expect(ottoListeningSummary(undefined)).toEqual(empty);
  });
});
