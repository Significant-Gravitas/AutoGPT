import * as Sentry from "@sentry/nextjs";
import { secondsBetween } from "./helpers";

// How long a call took to start, from the person's click to Otto's first
// audible sound, in five steps: the start answered and the microphone
// granted, both from the click, then the browser in the call's room, the
// relay's `ready`, and Otto's track carrying sound. Each mark is epoch ms, or
// null until the call gets there.
export interface CallMarks {
  clicked: number;
  granted: number | null;
  minted: number | null;
  joined: number | null;
  ready: number | null;
  firstSound: number | null;
}

// The marks of a call clicked at *clicked*.
export function newCallMarks(clicked: number): CallMarks {
  return {
    clicked,
    granted: null,
    minted: null,
    joined: null,
    ready: null,
    firstSound: null,
  };
}

// Seconds from the microphone's grant to each step the call has reached,
// for the call's `interview_started`: 0 for a step it had reached by then.
export function startSeconds(marks: CallMarks) {
  return {
    mint_s: since(marks, marks.minted),
    room_s: since(marks, marks.joined),
    ready_s: since(marks, marks.ready),
  };
}

// Sends call *callId*'s start to Sentry, the performance record: one span
// from the person's click to the latest of the page in the room, the relay's
// `ready` and Otto's first sound, with a span for each step under it. The relay can be ready,
// and Otto heard, before the page's room join settles, and Otto can be heard
// before the `ready` reaches the page; a step that came out of order lasts
// no time.
export function reportCallStart(callId: string, marks: CallMarks) {
  const { clicked, granted, minted, joined, ready, firstSound } = marks;
  if (granted === null || minted === null) return;
  if (joined === null || ready === null || firstSound === null) return;
  const start = Sentry.startInactiveSpan({
    name: "Voice call start",
    op: "voice.call.start",
    startTime: clicked,
    parentSpan: null,
    attributes: {
      call_id: callId,
      granted_s: secondsBetween(clicked, granted),
      mint_s: secondsBetween(granted, minted),
      room_s: secondsBetween(granted, joined),
      ready_s: secondsBetween(granted, ready),
      first_sound_s: secondsBetween(granted, firstSound),
    },
  });
  const steps: [string, number, number][] = [
    ["voice.call.grant", clicked, granted],
    ["voice.call.mint", clicked, minted],
    ["voice.call.room", Math.max(granted, minted), joined],
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
  start.end(Math.max(joined, ready, firstSound));
}

function since(marks: CallMarks, mark: number | null) {
  if (mark === null || marks.granted === null) return null;
  return secondsBetween(marks.granted, mark);
}
