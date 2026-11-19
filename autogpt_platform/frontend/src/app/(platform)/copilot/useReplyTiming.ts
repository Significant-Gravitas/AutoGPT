import { useMountEffect } from "@/hooks/useMountEffect";
import { useEffect, useInsertionEffect, useRef, type RefObject } from "react";
import {
  abandonReply,
  advanceReply,
  beginReplyTracking,
  type ReplyStep,
  type ReplyTimingStart,
  type ReplyTracking,
  type ReplyView,
  type StreamClient,
} from "./helpers/replyTiming";
import { recordReplySpan } from "./helpers/replyTimingSpan";
import type { SendInput } from "./stream/turnRuntime";

interface Args extends ReplyView {
  streamClient: StreamClient;
  sendMessage: (input: SendInput) => Promise<void>;
}

/**
 * Times each reply from the user's send to its first answer text on screen.
 * It reads the messages the page renders, so both stream clients are timed
 * the same way. Wraps the stream's `sendMessage`: a send given a start is
 * followed from the moment it is dispatched. Never throws into the chat.
 */
export function useReplyTiming({ streamClient, sendMessage, ...view }: Args) {
  const trackingRef = useRef<ReplyTracking | null>(null);
  const viewRef = useRef(view);
  const recheckTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useInsertionEffect(() => {
    viewRef.current = view;
  });

  function begin(timing: ReplyTimingStart) {
    updateReply(trackingRef, viewRef, recheckTimerRef, (tracking) =>
      abandonReply(tracking, performance.now()),
    );
    try {
      trackingRef.current = beginReplyTracking(
        timing,
        viewRef.current,
        streamClient,
      );
    } catch {
      trackingRef.current = null;
    }
  }

  // A send that returns without a turn ever showing was suppressed; one
  // whose turn was over before any render saw it running ends here too.
  function markReturned(timing: ReplyTimingStart) {
    if (trackingRef.current?.start !== timing) return;
    updateReply(trackingRef, viewRef, recheckTimerRef, (tracking) => ({
      tracking: { ...tracking, hasReturned: true },
      record: null,
      recheckInMs: 0,
    }));
  }

  function sendTimedMessage(input: SendInput, timing?: ReplyTimingStart) {
    if (timing) begin(timing);
    return dispatchAndSettle(sendMessage, input, () => {
      if (timing) markReturned(timing);
    });
  }

  useEffect(() => checkReply(trackingRef, viewRef, recheckTimerRef));
  // A chat switch unmounts the host and gives up its reply. Strict Mode's
  // development remount comes at once, so it cancels the give-up: a new
  // chat's first send is dispatched from this host's mount effect.
  const unmountTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useMountEffect(() => {
    if (unmountTimerRef.current) clearTimeout(unmountTimerRef.current);
    return () => {
      clearRecheck(recheckTimerRef);
      unmountTimerRef.current = setTimeout(() => {
        updateReply(trackingRef, viewRef, recheckTimerRef, (tracking) =>
          abandonReply(tracking, performance.now()),
        );
      }, 0);
    };
  });

  return { sendMessage: sendTimedMessage };
}

function updateReply(
  trackingRef: RefObject<ReplyTracking | null>,
  viewRef: RefObject<ReplyView>,
  recheckTimerRef: RefObject<ReturnType<typeof setTimeout> | null>,
  step: (tracking: ReplyTracking) => ReplyStep,
) {
  clearRecheck(recheckTimerRef);
  const tracking = trackingRef.current;
  if (!tracking) return;
  try {
    const { tracking: next, record, recheckInMs } = step(tracking);
    trackingRef.current = next;
    if (record) setTimeout(() => recordReplySpan(record), 0);
    if (next && recheckInMs !== undefined) {
      recheckTimerRef.current = setTimeout(() => {
        recheckTimerRef.current = null;
        checkReply(trackingRef, viewRef, recheckTimerRef);
      }, recheckInMs);
    }
  } catch {
    trackingRef.current = null;
  }
}

function checkReply(
  trackingRef: RefObject<ReplyTracking | null>,
  viewRef: RefObject<ReplyView>,
  recheckTimerRef: RefObject<ReturnType<typeof setTimeout> | null>,
) {
  updateReply(trackingRef, viewRef, recheckTimerRef, (tracking) =>
    advanceReply(tracking, viewRef.current, performance.now()),
  );
}

function clearRecheck(
  timerRef: RefObject<ReturnType<typeof setTimeout> | null>,
) {
  if (timerRef.current === null) return;
  clearTimeout(timerRef.current);
  timerRef.current = null;
}

async function dispatchAndSettle(
  sendMessage: Args["sendMessage"],
  input: SendInput,
  settled: () => void,
) {
  try {
    await sendMessage(input);
  } finally {
    settled();
  }
}
