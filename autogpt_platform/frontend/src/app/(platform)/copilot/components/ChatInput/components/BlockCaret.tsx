"use client";

import { useEffect, useState } from "react";

interface Props {
  textareaId: string;
}

// Style properties that must be mirrored from the textarea onto a hidden
// measurement div so that span.offsetTop/offsetLeft inside the mirror equal
// the caret's pixel position inside the textarea.
const MIRROR_PROPERTIES = [
  "direction",
  "boxSizing",
  "width",
  "height",
  "overflowX",
  "overflowY",
  "borderTopWidth",
  "borderRightWidth",
  "borderBottomWidth",
  "borderLeftWidth",
  "borderStyle",
  "paddingTop",
  "paddingRight",
  "paddingBottom",
  "paddingLeft",
  "fontStyle",
  "fontVariant",
  "fontWeight",
  "fontStretch",
  "fontSize",
  "fontSizeAdjust",
  "lineHeight",
  "fontFamily",
  "textAlign",
  "textTransform",
  "textIndent",
  "letterSpacing",
  "wordSpacing",
  "tabSize",
  "whiteSpace",
  "wordWrap",
  "wordBreak",
] as const;

interface Coords {
  left: number;
  top: number;
  height: number;
}

function getCaretCoordinates(
  textarea: HTMLTextAreaElement,
  position: number,
): Coords {
  const computed = window.getComputedStyle(textarea);
  const mirror = document.createElement("div");

  mirror.style.position = "absolute";
  mirror.style.top = "0";
  mirror.style.left = "-9999px";
  mirror.style.visibility = "hidden";
  mirror.style.whiteSpace = "pre-wrap";
  mirror.style.wordWrap = "break-word";

  for (const prop of MIRROR_PROPERTIES) {
    mirror.style[prop as never] = computed[prop as never];
  }

  mirror.textContent = textarea.value.substring(0, position);
  const marker = document.createElement("span");
  // Fall back to a real char so the span has a measurable bounding box even
  // when the cursor is at end-of-input.
  marker.textContent = textarea.value.substring(position) || ".";
  mirror.appendChild(marker);
  document.body.appendChild(mirror);

  const fontSize = parseFloat(computed.fontSize);
  const lineHeight = parseFloat(computed.lineHeight) || fontSize * 1.2;
  const coords: Coords = {
    left: marker.offsetLeft,
    top: marker.offsetTop,
    height: lineHeight,
  };

  document.body.removeChild(mirror);
  return coords;
}

const CARET_WIDTH = 3;

export function BlockCaret({ textareaId }: Props) {
  const [pos, setPos] = useState<Coords | null>(null);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const textarea = document.getElementById(
      textareaId,
    ) as HTMLTextAreaElement | null;
    if (!textarea) return;

    let frame = 0;

    function update() {
      if (!textarea) return;
      if (document.activeElement !== textarea) {
        setPos(null);
        return;
      }
      if (textarea.selectionStart !== textarea.selectionEnd) {
        // Hide bar during text selection — the bar over highlighted ranges
        // would look broken.
        setPos(null);
        return;
      }
      const c = getCaretCoordinates(textarea, textarea.selectionEnd);
      setPos({
        left: c.left - textarea.scrollLeft,
        top: c.top - textarea.scrollTop,
        height: c.height,
      });
    }

    // Each keystroke fires a burst of events (keydown + input + keyup, …).
    // Coalesce them into a single measurement per animation frame so the
    // synchronous reflow in getCaretCoordinates runs at most once per frame
    // instead of several times per keystroke.
    function scheduleUpdate() {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        update();
      });
    }

    update();
    const events = [
      "input",
      "keyup",
      "keydown",
      "click",
      "focus",
      "blur",
      "scroll",
      "select",
    ];
    for (const ev of events) textarea.addEventListener(ev, scheduleUpdate);
    window.addEventListener("resize", scheduleUpdate);

    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      for (const ev of events) textarea.removeEventListener(ev, scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, [textareaId]);

  useEffect(() => {
    if (!pos) return;
    // Reset to visible whenever the caret moves so typing keeps the bar
    // solid — only blinks once the user pauses.
    setVisible(true);
    const interval = setInterval(() => setVisible((v) => !v), 530);
    return () => clearInterval(interval);
  }, [pos]);

  if (!pos) return null;

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute z-10 bg-blue-500"
      style={{
        left: pos.left,
        top: pos.top - 3,
        height: pos.height,
        width: CARET_WIDTH,
        opacity: visible ? 1 : 0,
      }}
    />
  );
}
