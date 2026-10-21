import type { MentionInput } from "../ChatInput/useChatMentions";

/** One serialized run of the editor DOM: a text node, a badge (which
 *  serializes as its credential token), or a line break. A break is either a
 *  BR or a block element whose start implies a newline after its predecessor.
 *  A BR that is the last child of its block is the placeholder browsers keep
 *  so an empty line has height; it is not a newline of its own. */
interface Piece {
  node: Node;
  text: string;
  kind: "text" | "token" | "break";
}

function isBlock(node: Node) {
  return node.nodeName === "DIV" || node.nodeName === "P";
}

function editorPieces(node: Node, pieces: Piece[] = []): Piece[] {
  if (node instanceof HTMLElement && node.dataset.credentialMention) {
    pieces.push({ node, text: node.dataset.credentialMention, kind: "token" });
    return pieces;
  }
  if (node.nodeType === Node.TEXT_NODE) {
    pieces.push({ node, text: node.textContent ?? "", kind: "text" });
    return pieces;
  }
  if (node.nodeName === "BR") {
    if (node.nextSibling) pieces.push({ node, text: "\n", kind: "break" });
    return pieces;
  }
  let previous: Node | null = null;
  for (const child of Array.from(node.childNodes)) {
    if (previous && (isBlock(child) || isBlock(previous)))
      pieces.push({ node: child, text: "\n", kind: "break" });
    editorPieces(child, pieces);
    previous = child;
  }
  return pieces;
}

export function readMentionEditor(node: Node): string {
  return editorPieces(node)
    .map((piece) => piece.text)
    .join("");
}

/** Serialized text runs between badges, so a credential reference typed or
 *  pasted as plain text can be found even when the browser split it across
 *  text nodes or lines. */
export function editorTextRuns(root: HTMLElement): string[] {
  const runs: string[] = [];
  let current = "";
  for (const piece of editorPieces(root)) {
    if (piece.kind === "token") {
      runs.push(current);
      current = "";
    } else current += piece.text;
  }
  runs.push(current);
  return runs;
}

function pointOffset(root: HTMLElement, container: Node, offset: number) {
  const point = document.createRange();
  point.setStart(container, offset);
  point.collapse(true);
  let position = 0;
  for (const piece of editorPieces(root)) {
    if (piece.kind === "text" && piece.node === container)
      return position + Math.min(offset, piece.text.length);
    if (point.comparePoint(piece.node, 0) === 1) return position;
    position += piece.text.length;
  }
  return position;
}

export function editorSelectionRange(root: HTMLElement) {
  const selection = window.getSelection();
  if (!selection?.rangeCount) return null;
  const range = selection.getRangeAt(0);
  if (
    !root.contains(range.startContainer) ||
    !root.contains(range.endContainer)
  )
    return null;
  return {
    start: pointOffset(root, range.startContainer, range.startOffset),
    end: pointOffset(root, range.endContainer, range.endOffset),
  };
}

export function editorSelectionOffset(root: HTMLElement) {
  return editorSelectionRange(root)?.start ?? readMentionEditor(root).length;
}

function pointAt(root: HTMLElement, offset: number): Range {
  const range = document.createRange();
  let position = 0;
  for (const piece of editorPieces(root)) {
    const end = position + piece.text.length;
    if (piece.kind === "text" && offset <= end) {
      range.setStart(piece.node, Math.max(0, offset - position));
      return range;
    }
    if (piece.kind === "token" && offset === position) {
      range.setStartBefore(piece.node);
      return range;
    }
    if (piece.kind === "token" && offset < end) {
      range.setStartAfter(piece.node);
      return range;
    }
    if (piece.kind === "break" && offset === position) {
      range.setStartBefore(piece.node);
      return range;
    }
    if (piece.kind === "break" && offset === end) {
      if (piece.node.nodeName === "BR") range.setStartAfter(piece.node);
      else range.setStart(piece.node, 0);
      return range;
    }
    position = end;
  }
  range.selectNodeContents(root);
  range.collapse(false);
  return range;
}

export function setEditorCaret(root: HTMLElement, start: number, end = start) {
  const from = pointAt(root, Math.max(0, start));
  const to = pointAt(root, Math.max(start, end));
  const range = document.createRange();
  range.setStart(from.startContainer, from.startOffset);
  range.setEnd(to.startContainer, to.startOffset);
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(range);
}

export function mentionInputFor(root: HTMLElement): MentionInput {
  return {
    get value() {
      return readMentionEditor(root);
    },
    get selectionStart() {
      return editorSelectionOffset(root);
    },
    setSelectionRange(start: number, end: number) {
      setEditorCaret(root, start, end);
    },
  };
}
