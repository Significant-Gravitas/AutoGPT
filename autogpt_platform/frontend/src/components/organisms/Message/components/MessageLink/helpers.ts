import type { MouseEvent } from "react";

// Streamdown's href for a link whose markdown is still streaming in.
export const INCOMPLETE_LINK_HREF = "streamdown:incomplete-link";

const LEFT_BUTTON = 0;
const MIDDLE_BUTTON = 1;

// Schemes a link to another site may open, after the dialog. Streamdown's
// sanitising already drops the rest; this keeps it so if that ever changes.
const OPENABLE_PROTOCOLS = ["http:", "https:", "mailto:"];

type LinkKind = "site" | "external" | "inert";

// A path on this site, known without `window`, so server and browser agree.
function isSitePath(href: string) {
  return /^\/(?![/\\])/.test(href) || /^[#?]/.test(href);
}

// `origin` is null while the page is server-rendered or hydrating; an absolute
// link to this site is then drawn as external until the browser's origin is in.
export function kindOfLink(
  href: string | undefined,
  origin: string | null,
): LinkKind {
  if (!href || href === INCOMPLETE_LINK_HREF) return "inert";
  if (isSitePath(href)) return "site";
  try {
    const url = origin ? new URL(href, origin) : new URL(href);
    if (url.origin === origin) return "site";
    return OPENABLE_PROTOCOLS.includes(url.protocol) ? "external" : "inert";
  } catch {
    return "inert";
  }
}

// Every way of opening a link to another site asks first (SECRT-2096): a left
// click, with or without Ctrl, Cmd, Shift or Alt, and a middle-click. The right
// button only opens the context menu, which offers no "Open in new tab" for a
// button. On macOS a Ctrl-click is a right-click, so it opens that menu too.
export function asksBeforeOpening(event: MouseEvent) {
  return event.button === LEFT_BUTTON || event.button === MIDDLE_BUTTON;
}

// A middle press starts the browser's autoscroll on some systems, which can
// swallow the auxclick that asks.
export function isMiddlePress(event: MouseEvent) {
  return event.button === MIDDLE_BUTTON;
}
