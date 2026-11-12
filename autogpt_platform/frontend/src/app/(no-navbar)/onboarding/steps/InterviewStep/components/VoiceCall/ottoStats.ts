// How Otto's speech fared on its way to the person, from WebRTC's stats for
// his track in the call's room: the share of his audio the browser had to
// make up for packets that never came, the packets lost, and the worst
// jitter and round trip seen. Otto is silent for much of a call, and what the
// browser fills a silent stretch with is counted apart, so the share of his
// speech made up is not the share of his silence. Each track counts from
// nothing, and a rejoin or a LiveKit reconnect brings a new one, so each
// stream's latest counts are kept apart and summed.
export interface OttoListening {
  streams: Map<string, StreamCounts>;
  jitterMsMax: number | null;
  rttMsMax: number | null;
}

interface StreamCounts {
  concealedSamples: number;
  // The concealed samples that filled silence, a part of concealedSamples.
  silentConcealedSamples: number;
  totalSamplesReceived: number;
  packetsLost: number;
}

type Stat = Record<string, unknown>;

export function newOttoListening(): OttoListening {
  return { streams: new Map(), jitterMsMax: null, rttMsMax: null };
}

// Folds one stats report of Otto's track into *listening*.
export function addOttoStats(listening: OttoListening, report: RTCStatsReport) {
  const stats = statsIn(report);
  stats.forEach((stat) => {
    if (stat.type !== "inbound-rtp" || stat.kind !== "audio") return;
    if (typeof stat.id !== "string") return;
    listening.streams.set(stat.id, {
      concealedSamples: countOf(stat.concealedSamples),
      silentConcealedSamples: countOf(stat.silentConcealedSamples),
      totalSamplesReceived: countOf(stat.totalSamplesReceived),
      packetsLost: countOf(stat.packetsLost),
    });
    listening.jitterMsMax = highest(listening.jitterMsMax, msOf(stat.jitter));
  });
  const pair = pairInUse(stats);
  if (pair) {
    listening.rttMsMax = highest(
      listening.rttMsMax,
      msOf(pair.currentRoundTripTime),
    );
  }
}

// What a call's `interview_ended` carries of it: null for what was never
// read, as for a call that ended before Otto spoke.
export function ottoListeningSummary(listening: OttoListening | undefined) {
  const streams = [...(listening?.streams.values() ?? [])];
  const received = sumOf(streams, (stream) => stream.totalSamplesReceived);
  const concealed = sumOf(streams, (stream) => stream.concealedSamples);
  const silent = sumOf(streams, (stream) =>
    Math.min(stream.silentConcealedSamples, stream.concealedSamples),
  );
  return {
    concealed_share: shareOf(concealed - silent, received),
    silent_concealed_share: shareOf(silent, received),
    packets_lost:
      streams.length > 0
        ? sumOf(streams, (stream) => stream.packetsLost)
        : null,
    jitter_ms_max: listening?.jitterMsMax ?? null,
    rtt_ms_max: listening?.rttMsMax ?? null,
  };
}

function statsIn(report: RTCStatsReport) {
  const stats: Stat[] = [];
  report.forEach((value: unknown) => {
    if (isStat(value)) stats.push(value);
  });
  return stats;
}

function isStat(value: unknown): value is Stat {
  return typeof value === "object" && value !== null;
}

// The candidate pair carrying the call's media: the one its transport names,
// or, in Firefox, the one marked selected.
function pairInUse(stats: Stat[]) {
  const pairs = stats.filter((stat) => stat.type === "candidate-pair");
  const transport = stats.find(
    (stat) =>
      stat.type === "transport" &&
      typeof stat.selectedCandidatePairId === "string",
  );
  if (transport) {
    return pairs.find((pair) => pair.id === transport.selectedCandidatePairId);
  }
  return pairs.find((pair) => pair.selected === true);
}

function shareOf(part: number, whole: number) {
  return whole > 0 ? Math.round((part / whole) * 10_000) / 10_000 : null;
}

function countOf(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, value)
    : 0;
}

function msOf(seconds: unknown) {
  return typeof seconds === "number" && Number.isFinite(seconds)
    ? Math.round(seconds * 1_000)
    : null;
}

function highest(current: number | null, next: number | null) {
  if (next === null) return current;
  return current === null ? next : Math.max(current, next);
}

function sumOf(
  streams: StreamCounts[],
  count: (stream: StreamCounts) => number,
) {
  return streams.reduce((total, stream) => total + count(stream), 0);
}
