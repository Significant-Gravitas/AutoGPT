import { css } from "@codemirror/lang-css";
import { html } from "@codemirror/lang-html";
import { javascript } from "@codemirror/lang-javascript";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { python } from "@codemirror/lang-python";
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

/** Neutral CodeMirror theme that blends into the light `bg-sidebar` panel. */
export const editorTheme: Extension = EditorView.theme({
  "&": {
    backgroundColor: "transparent",
    color: "#3f3f46",
    fontSize: "13px",
  },
  ".cm-content": {
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  },
  ".cm-gutters": {
    backgroundColor: "transparent",
    color: "#a1a1aa",
    border: "none",
  },
  "&.cm-focused": { outline: "none" },
  ".cm-activeLine": { backgroundColor: "rgba(0, 0, 0, 0.03)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": {
    backgroundColor: "#e4e4e7",
  },
});

/** xterm theme — dark terminal with a standard ANSI palette. */
export const xtermTheme = {
  background: "#18181b",
  foreground: "#e4e4e7",
  cursor: "#e4e4e7",
  cursorAccent: "#18181b",
  selectionBackground: "#3f3f46",
  black: "#18181b",
  red: "#f87171",
  green: "#4ade80",
  yellow: "#facc15",
  blue: "#60a5fa",
  magenta: "#c084fc",
  cyan: "#22d3ee",
  white: "#e4e4e7",
  brightBlack: "#52525b",
  brightRed: "#fca5a5",
  brightGreen: "#86efac",
  brightYellow: "#fde047",
  brightBlue: "#93c5fd",
  brightMagenta: "#d8b4fe",
  brightCyan: "#67e8f9",
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
