"use client";

import { memo, useLayoutEffect, useRef, useState } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";

interface Props {
  messageID: string | null;
  bottomInset: number;
}

// Breathing room above the composer so the pinned turn does not sit flush
// against the bottom inset.
const TAIL_GAP = 24;

// Share of the viewport the newest turn is allowed to occupy. The turn is
// bottom-aligned, so this is also where the user's message lands: 0.5 parks
// it mid-screen, 1 would pin it to the very top.
const TAIL_VIEWPORT_RATIO = 0.5;

/** Reserves room below the newest turn so that sticking to the bottom parks
 *  the user's message around the middle of the viewport, leaving the lower
 *  half of the screen for the reply. Collapses to zero once the reply is tall
 *  enough to fill that space, handing scrolling back to StickToBottom. */
export const TailSpacer = memo(function TailSpacer({
  messageID,
  bottomInset,
}: Props) {
  const { scrollRef } = useStickToBottomContext();
  const spacerRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(0);
  // The observer fires on every resize of the transcript; only a new height
  // is worth a render.
  const heightRef = useRef(0);

  function applyHeight(next: number) {
    if (next === heightRef.current) return;
    heightRef.current = next;
    setHeight(next);
  }

  useLayoutEffect(() => {
    const scroller = scrollRef.current;
    const spacer = spacerRef.current;
    // The spacer is the last child of the scroll content, so its parent is
    // the element whose growth we care about.
    const content = spacer?.parentElement;
    if (!scroller || !content || !spacer || !messageID) {
      applyHeight(0);
      return;
    }
    const selector = `[data-message-id="${CSS.escape(messageID)}"]`;
    if (!content.querySelector(selector)) applyHeight(0);

    function measure() {
      if (!scroller || !spacer || !content) return;
      // Looked up on every pass: the windowed transcript unmounts a row that
      // scrolls far out of view and mounts a new node when it comes back.
      // While it is away the last height stands.
      const message = content.querySelector(selector);
      if (!(message instanceof HTMLElement)) return;
      // Measured against the spacer's own top, which sits above its height —
      // reading the content box instead would feed this back into itself.
      const turnHeight =
        spacer.getBoundingClientRect().top -
        message.getBoundingClientRect().top;
      const viewport = scroller.clientHeight - bottomInset - TAIL_GAP;
      applyHeight(Math.max(0, viewport * TAIL_VIEWPORT_RATIO - turnHeight));
    }

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(content);
    observer.observe(scroller);
    return () => observer.disconnect();
  }, [messageID, bottomInset, scrollRef]);

  // `-mt-6` cancels the parent's `gap-6` so zero height really is zero —
  // without it every thread keeps an extra gap above the composer.
  return (
    <div ref={spacerRef} aria-hidden className="-mt-6" style={{ height }} />
  );
});
