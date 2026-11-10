import { splitTextSegments } from "@/lib/text-segments";

/**
 * Marks up a reply while it streams in, for the `[data-stream-*]` rules in
 * `globals.css`: each word gets its own span, so a word that arrives mounts
 * a new span whose fade plays once (React keeps the spans already shown),
 * and the end of the text gets an empty caret span. Streamdown renders each
 * block on its own, so every block gets a caret; `STREAMING_CLASS` shows only
 * the last block's, and no caret shows without it.
 *
 * Streamdown's own elements re-render only when their class or source
 * position changes, so every element on the way down to the caret is marked
 * with `CARET_PATH_CLASS`: when the stream ends and the plugin is gone, the
 * mark goes and those elements render again, without the caret. Words
 * elsewhere may keep their finished spans; they look the same as text.
 */

interface HastNode {
  type: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

// Text inside these is read whole by its renderer (code, maths, diagrams) or
// is not prose.
const UNWRAPPED_TAGS = new Set([
  "pre",
  "code",
  "kbd",
  "svg",
  "math",
  "script",
  "style",
]);
// Elements the caret can sit inside, at the end of their text.
const CARET_HOSTS = new Set([
  "p",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "li",
  "ul",
  "ol",
  "blockquote",
  "strong",
  "em",
  "b",
  "i",
  "a",
  "del",
  "s",
  "u",
  "mark",
  "sup",
  "sub",
]);
// Inline elements the caret can follow.
const INLINE_LEAVES = new Set(["span", "code", "kbd", "img", "br"]);

/** Shows the last block's caret (`@utility` in globals.css). */
export const STREAMING_CLASS = "streaming-reply";
const CARET_PATH_CLASS = "streaming-caret-path";

export function rehypeStreamReveal() {
  return function revealStreamingText(tree: HastNode) {
    wrapWords(tree);
    const path = caretPath(tree);
    if (path.length === 0) return;
    for (const element of path) markCaretPath(element);
    path[path.length - 1].children?.push(caret());
  };
}

function wrapWords(parent: HastNode) {
  if (!parent.children) return;
  parent.children = parent.children.flatMap((child) => {
    if (child.type === "element") {
      if (!UNWRAPPED_TAGS.has(child.tagName ?? "")) wrapWords(child);
      return [child];
    }
    if (child.type !== "text" || !child.value?.trim()) return [child];
    return splitTextSegments(child.value).map(word);
  });
}

function word(value: string): HastNode {
  return {
    type: "element",
    tagName: "span",
    properties: { dataStreamFade: "" },
    children: [{ type: "text", value }],
  };
}

function caret(): HastNode {
  return {
    type: "element",
    tagName: "span",
    properties: { dataStreamCaret: "", ariaHidden: "true" },
    children: [],
  };
}

function markCaretPath(element: HastNode) {
  const properties = (element.properties ??= {});
  const className = properties.className;
  properties.className = Array.isArray(className)
    ? [...className, CARET_PATH_CLASS]
    : [CARET_PATH_CLASS];
}

// The elements down to the innermost one whose content ends the block, if
// the caret can sit on its last line: not after a code block, a table or
// other block content. Empty when it cannot.
function caretPath(root: HastNode): HastNode[] {
  const path: HastNode[] = [];
  let current = root;
  for (;;) {
    const last = lastContent(current);
    if (!last) return [];
    if (last.type === "text" || INLINE_LEAVES.has(last.tagName ?? "")) {
      return path;
    }
    if (last.type !== "element" || !CARET_HOSTS.has(last.tagName ?? "")) {
      return [];
    }
    path.push(last);
    current = last;
  }
}

function lastContent(node: HastNode): HastNode | undefined {
  const children = node.children ?? [];
  for (let i = children.length - 1; i >= 0; i--) {
    const child = children[i];
    if (child.type === "text" && !child.value?.trim()) continue;
    if (child.type === "comment") continue;
    return child;
  }
  return undefined;
}
