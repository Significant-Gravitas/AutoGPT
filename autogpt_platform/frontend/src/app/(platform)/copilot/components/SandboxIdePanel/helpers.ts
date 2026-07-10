import { css } from "@codemirror/lang-css";
import { html } from "@codemirror/lang-html";
import { javascript } from "@codemirror/lang-javascript";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { python } from "@codemirror/lang-python";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags as t } from "@lezer/highlight";
import { EditorView, type Extension } from "@uiw/react-codemirror";

/** Pick CodeMirror language extensions from a file extension. */
export function getLanguageExtension(path: string): Extension[] {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  switch (ext) {
    case "ts":
    case "tsx":
      return [javascript({ jsx: true, typescript: true })];
    case "js":
    case "jsx":
    case "mjs":
    case "cjs":
      return [javascript({ jsx: true })];
    case "py":
      return [python()];
    case "json":
      return [json()];
    case "md":
    case "markdown":
      return [markdown()];
    case "html":
    case "htm":
      return [html()];
    case "css":
      return [css()];
    default:
      return [];
  }
}

/** Atom One Light–style token colors for readable syntax highlighting. */
const highlightStyle = HighlightStyle.define([
  { tag: [t.keyword, t.moduleKeyword, t.operatorKeyword], color: "#a626a4" },
  { tag: [t.controlKeyword, t.definitionKeyword], color: "#a626a4" },
  { tag: [t.string, t.special(t.string), t.inserted], color: "#50a14f" },
  { tag: [t.comment, t.meta], color: "#a0a1a7", fontStyle: "italic" },
  { tag: [t.number, t.bool, t.atom, t.null], color: "#986801" },
  {
    tag: [t.function(t.variableName), t.function(t.propertyName)],
    color: "#4078f2",
  },
  { tag: [t.typeName, t.className, t.namespace], color: "#c18401" },
  { tag: [t.propertyName, t.attributeName], color: "#e45649" },
  { tag: [t.operator, t.punctuation, t.separator], color: "#383a42" },
  { tag: [t.tagName, t.heading], color: "#e45649" },
  { tag: [t.constant(t.variableName), t.standard(t.name)], color: "#986801" },
  { tag: [t.regexp, t.escape, t.link], color: "#0184bc" },
  { tag: t.invalid, color: "#e45649" },
  { tag: t.strong, fontWeight: "bold" },
  { tag: t.emphasis, fontStyle: "italic" },
]);

/** Syntax highlighting extension for the file editor and diff views. */
export const codeHighlighting: Extension = syntaxHighlighting(highlightStyle);

/** Neutral CodeMirror theme that blends into the white sandbox panel. */
export const editorTheme: Extension = EditorView.theme({
  "&": {
    backgroundColor: "#ffffff",
    color: "#3f3f46",
    fontSize: "13px",
  },
  ".cm-content": {
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  },
  ".cm-gutters": {
    backgroundColor: "#ffffff",
    color: "#a1a1aa",
    border: "none",
  },
  "&.cm-focused": { outline: "none" },
  ".cm-activeLine": { backgroundColor: "rgba(0, 0, 0, 0.03)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": {
    backgroundColor: "#e4e4e7",
  },
  // Left-of-line "comment" button on the hovered line — see lineCommentButton.ts
  ".cm-add-comment-anchor": {
    position: "relative",
    display: "inline-block",
    width: "0",
    height: "0",
    verticalAlign: "text-top",
  },
  ".cm-add-comment-button": {
    position: "absolute",
    // Sit above the line-number gutter (CodeMirror gives .cm-gutters z-index
    // 200), otherwise this button — which overhangs into the gutter — is
    // painted behind the numbers.
    zIndex: "300",
    left: "-1.6rem",
    top: "-0.15rem",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "1.375rem",
    height: "1.375rem",
    padding: "0",
    border: "none",
    borderRadius: "0.625rem",
    cursor: "pointer",
    fontSize: "16px",
    fontWeight: "600",
    lineHeight: "1",
    color: "hsl(var(--primary-foreground))",
    backgroundColor: "hsl(var(--primary))",
    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.2)",
  },
  ".cm-add-comment-button:hover": {
    opacity: "0.9",
  },
});

/** xterm theme — light terminal on white with a standard ANSI palette. */
export const xtermTheme = {
  background: "#ffffff",
  foreground: "#27272a",
  cursor: "#27272a",
  cursorAccent: "#ffffff",
  selectionBackground: "#e4e4e7",
  black: "#27272a",
  red: "#dc2626",
  green: "#16a34a",
  yellow: "#ca8a04",
  blue: "#2563eb",
  magenta: "#9333ea",
  cyan: "#0891b2",
  white: "#e4e4e7",
  brightBlack: "#52525b",
  brightRed: "#ef4444",
  brightGreen: "#22c55e",
  brightYellow: "#eab308",
  brightBlue: "#3b82f6",
  brightMagenta: "#a855f7",
  brightCyan: "#06b6d4",
  brightWhite: "#fafafa",
};

/** Last path segment of a workspace path. */
export function basename(path: string): string {
  const segments = path.split("/").filter(Boolean);
  return segments[segments.length - 1] || path;
}

/** Build the terminal WebSocket URL from the REST API base + auth token. */
export function buildTerminalWsUrl(args: {
  restApiUrl: string;
  sessionId: string;
  token: string;
}): string {
  const { restApiUrl, sessionId, token } = args;
  const wsBase = restApiUrl.replace(/^http/, "ws");
  return `${wsBase}/chat/sessions/${sessionId}/sandbox/terminal?token=${encodeURIComponent(
    token,
  )}`;
}

/** Map a git status letter to its badge label. */
export const STATUS_LABELS: Record<string, string> = {
  M: "M",
  A: "A",
  D: "D",
  R: "R",
  "?": "?",
};
