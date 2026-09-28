import { describe, expect, it } from "vitest";
import { parseCredentialMentions } from "../helpers";
import {
  editorSelectionOffset,
  readMentionEditor,
  setEditorCaret,
} from "../editorHelpers";

const BADGE = `<span contenteditable="false" data-credential-mention="[Work](credential://google/work-id)"></span>`;

function editorWith(html: string) {
  const root = document.createElement("div");
  root.contentEditable = "true";
  root.innerHTML = html;
  document.body.appendChild(root);
  return root;
}

describe("editor newline serialization", () => {
  it.each([
    ["Hello<div><br></div>", "Hello\n"],
    ["First<br>Second", "First\nSecond"],
    ["<div>First</div><div>Second</div>", "First\nSecond"],
    ["<div>A</div><div><br></div><div>B</div>", "A\n\nB"],
    ["<p>A</p><p>B</p>", "A\nB"],
    ["A<br><br>", "A\n"],
    ["<br>", ""],
    [
      `Check ${BADGE}<div>next</div>`,
      "Check [Work](credential://google/work-id)\nnext",
    ],
  ])("serializes %s as %j", (html, expected) => {
    expect(readMentionEditor(editorWith(html))).toBe(expected);
  });
});

function snapPastBadge(text: string, offset: number) {
  let position = 0;
  for (const part of parseCredentialMentions(text)) {
    const length = typeof part === "string" ? part.length : part.token.length;
    if (
      typeof part !== "string" &&
      offset > position &&
      offset < position + length
    )
      return position + length;
    position += length;
  }
  return offset;
}

describe("editor caret round trip", () => {
  it.each([
    "First<br>Second",
    "<div>First</div><div>Second</div>",
    "<div>A</div><div><br></div><div>B</div>",
    "<p>A</p><p>B</p>",
    "Hello<div><br></div>",
    `Check ${BADGE} then<div>next ${BADGE}</div>`,
  ])("reads back every caret offset in %s", (html) => {
    const root = editorWith(html);
    const text = readMentionEditor(root);
    for (let offset = 0; offset <= text.length; offset++) {
      setEditorCaret(root, offset);
      expect(editorSelectionOffset(root), `offset ${offset}`).toBe(
        snapPastBadge(text, offset),
      );
    }
  });

  it("selects a serialized range across a badge", () => {
    const root = editorWith(`Check ${BADGE} then`);
    setEditorCaret(
      root,
      2,
      "Check [Work](credential://google/work-id) th".length,
    );
    const range = window.getSelection()!.getRangeAt(0);
    expect(range.collapsed).toBe(false);
    expect(range.toString()).toBe("eck  th");
  });
});
