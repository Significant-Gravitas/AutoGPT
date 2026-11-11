import * as Sentry from "@sentry/nextjs";
import { secondsBetween } from "./helpers";

// How long a call took to start, from the microphone being granted to Otto's
// first audible sound, in four steps: the start answered, the browser in the
// call's room, the relay's `ready`, and Otto's track carrying sound. Each mark
// is epoch ms, or null until the call gets there.
export interface CallMarks {
  granted: number;
  minted: number | null;
  joined: number | null;
  ready: number | null;
  firstSound: number | null;
}

export function newCallMarks(): CallMarks {
  return {
    granted: Date.now(),
    minted: null,
    joined: null,
    ready: null,
    firstSound: null,
  };
}

// Seconds from the microphone to each step the call has reached, for the
// call's `interview_started`.
export function startSeconds(marks: CallMarks) {
  return {
    mint_s: since(marks, marks.minted),
    room_s: since(marks, marks.joined),
    ready_s: since(marks, marks.ready),
  };
}

// Sends call *callId*'s start to Sentry, the performance record: one span
// from the microphone to the later of the relay's `ready` and Otto's first
// sound, with a span for each step under it. Otto can be heard before the
// `ready` reaches the page; his first-sound step then lasts no time.
export function reportCallStart(callId: string, marks: CallMarks) {
  const { granted, minted, joined, ready, firstSound } = marks;
  if (minted === null || joined === null || ready === null) return;
  if (firstSound === null) return;
  const start = Sentry.startInactiveSpan({
    name: "Voice call start",
    op: "voice.call.start",
    startTime: granted,
    parentSpan: null,
    attributes: {
      call_id: callId,
      mint_s: secondsBetween(granted, minted),
      room_s: secondsBetween(granted, joined),
      ready_s: secondsBetween(granted, ready),
      first_sound_s: secondsBetween(granted, firstSound),
    },
  });
  const steps: [string, number, number][] = [
    ["voice.call.mint", granted, minted],
    ["voice.call.room", minted, joined],
    ["voice.call.ready", joined, ready],
    ["voice.call.first_sound", ready, firstSound],
  ];
  steps.forEach(([op, from, to]) => {
    Sentry.startInactiveSpan({
      name: op,
      op,
      startTime: from,
      parentSpan: start,
    }).end(Math.max(from, to));
  });
  start.end(Math.max(ready, firstSound));
}

function since(marks: CallMarks, mark: number | null) {
  return mark === null ? null : secondsBetween(marks.granted, mark);
}
