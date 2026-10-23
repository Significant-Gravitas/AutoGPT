import { useMountEffect } from "@/hooks/useMountEffect";
import { useEffect, useRef } from "react";
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
  viewRef.current = view;

  function update(step: (tracking: ReplyTracking) => ReplyStep) {
    const tracking = trackingRef.current;
    if (!tracking) return;
    try {
      const { tracking: next, record, recheckInMs } = step(tracking);
      trackingRef.current = next;
      // Reported off the render path, so a reply never waits on Sentry.
      if (record) setTimeout(() => recordReplySpan(record), 0);
      if (recheckInMs) setTimeout(checkReply, recheckInMs);
    } catch {
      trackingRef.current = null;
    }
  }

  function checkReply() {
    update((tracking) =>
      advanceReply(tracking, viewRef.current, performance.now()),
    );
  }

  function begin(timing: ReplyTimingStart) {
    update((tracking) => abandonReply(tracking, performance.now()));
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
    update((tracking) => ({
      tracking:
        tracking.start === timing
          ? { ...tracking, hasReturned: true }
          : tracking,
      record: null,
    }));
    setTimeout(checkReply, 0);
  }

  async function sendTimedMessage(input: SendInput, timing?: ReplyTimingStart) {
    if (timing) begin(timing);
    try {
      await sendMessage(input);
    } finally {
      if (timing) markReturned(timing);
    }
  }

  useEffect(checkReply);
  // A chat switch unmounts the host and gives up its reply. Strict Mode's
  // development remount comes at once, so it cancels the give-up: a new
  // chat's first send is dispatched from this host's mount effect.
  const unmountTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useMountEffect(() => {
    if (unmountTimerRef.current) clearTimeout(unmountTimerRef.current);
    return () => {
      unmountTimerRef.current = setTimeout(() => {
        update((tracking) => abandonReply(tracking, performance.now()));
      }, 0);
    };
  });

  return { sendMessage: sendTimedMessage };
}
