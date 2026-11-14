import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { PageTranslatorEvent } from "@/services/analytics/posthog-events";

type SkippedCall = "removeChild" | "insertBefore";

let skipReported = false;

/**
 * A page translator swaps React's text nodes for its own `<font>` elements, so
 * React's next `removeChild` or `insertBefore` there names a node that is no
 * longer a child and the DOM throws (facebook/react#11538). Skip those calls.
 */
export function toleratePageTranslators() {
  const { removeChild, insertBefore } = Node.prototype;

  Node.prototype.removeChild = function <T extends Node>(
    this: Node,
    child: T,
  ): T {
    if (isMisplaced(child, this)) {
      reportSkip("removeChild");
      return child;
    }
    return removeChild.call(this, child) as T;
  };

  Node.prototype.insertBefore = function <T extends Node>(
    this: Node,
    node: T,
    child: Node | null,
  ): T {
    if (isMisplaced(child, this)) {
      reportSkip("insertBefore");
      return node;
    }
    return insertBefore.call(this, node, child) as T;
  };
}

function isMisplaced(node: Node | null, parent: Node) {
  return node instanceof Node && node.parentNode !== parent;
}

// No console output: captureConsoleIntegration would send it to Sentry.
// Deferred so the commit React is in the middle of never runs analytics code.
function reportSkip(call: SkippedCall) {
  if (skipReported) return;
  skipReported = true;
  queueMicrotask(() =>
    capturePostHogEvent(PageTranslatorEvent.DOM_CALL_SKIPPED, {
      call,
      path: window.location.pathname,
    }),
  );
}
