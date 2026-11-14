import type { UIMessage } from "ai";
import { parseAnswersFrame } from "../../QuestionDock/answeredQuestions";
import { getLatestTaskList } from "../../TaskProgressBar/helpers";
import { toChainRow } from "../../ToolChain/helpers";

export interface OverviewStep {
  key: string;
  text: string;
  state: "done" | "now" | "later";
}

export interface ChatUsage {
  searches: number;
  pagesRead: number;
  skills: string[];
  /** Provider slugs of the apps Otto used here, e.g. "gmail". */
  connectors: string[];
  /** Ids of the experts work was handed to or who were consulted. */
  teammateIDs: string[];
  memoryChecks: number;
  notesSaved: number;
  /** Commands run on the chat's computer. */
  commands: number;
}

export const STEPS_SHOWN = 3;

interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output: unknown;
  /** The capability id when the call went through run_capability. */
  capability: string | null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function text(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/** Every tool call of the chat, with a run_capability call read as the tool
 *  behind it. */
function toolCallsOf(messages: UIMessage[]): ToolCall[] {
  return messages.flatMap((message) =>
    message.role !== "assistant"
      ? []
      : message.parts.flatMap((part): ToolCall[] => {
          if (!part.type.startsWith("tool-")) return [];
          const raw = part as {
            type: string;
            input?: unknown;
            output?: unknown;
          };
          const outer = raw.type.slice("tool-".length);
          const input = asRecord(raw.input);
          if (outer !== "run_capability") {
            return [
              { name: outer, input, output: raw.output, capability: null },
            ];
          }
          const id = text(input.id) ?? "";
          const inner = asRecord(input.input);
          return id.startsWith("tool:")
            ? [
                {
                  name: id.slice("tool:".length),
                  input: inner,
                  output: raw.output,
                  capability: null,
                },
              ]
            : [{ name: id, input: inner, output: raw.output, capability: id }];
        }),
  );
}

/** The app a call went through: the provider it names, or the server in an
 *  `mcp:<server>/<tool>` id. */
function providerOf(call: ToolCall) {
  const named =
    text(call.input.provider) ?? text(asRecord(call.output).provider);
  if (named) return named.toLowerCase();
  if (!call.capability?.startsWith("mcp:")) return null;
  const server = call.capability.slice("mcp:".length).split("/")[0];
  return server ? server.toLowerCase() : null;
}

function unique(values: (string | null)[]) {
  return [...new Set(values.filter((value): value is string => !!value))];
}

/** What this chat used, read from its tool calls. */
export function chatUsageOf(messages: UIMessage[]): ChatUsage {
  const calls = toolCallsOf(messages);
  function count(name: string) {
    return calls.filter((call) => call.name === name).length;
  }
  return {
    searches: count("web_search"),
    pagesRead: count("web_fetch") + count("browser_navigate"),
    skills: unique(
      calls
        .filter((call) => call.name === "read_skill")
        .map(
          (call) =>
            text(call.input.name) ??
            text(call.input.skill_name) ??
            text(call.input.skill),
        ),
    ),
    connectors: unique(calls.map(providerOf)),
    teammateIDs: unique(
      calls
        .filter((call) =>
          [
            "delegate_to_expert",
            "consult_teammate",
            "handoff_to_expert",
            "start_task",
          ].includes(call.name),
        )
        .map((call) => text(call.input.expert_id)),
    ),
    memoryChecks: count("memory_search"),
    notesSaved: count("memory_store"),
    commands: count("bash_exec") + count("Bash"),
  };
}

/** The steps of the turn in progress or just finished: Otto's own checklist
 *  when it wrote one, else what it did, in the words the chat shows. */
export function turnStepsOf(messages: UIMessage[]): OverviewStep[] {
  const todos = getLatestTaskList(messages);
  if (todos.length > 0) {
    return todos.map((todo, index) => ({
      key: `todo-${index}`,
      text: todo.content,
      state:
        todo.status === "completed"
          ? "done"
          : todo.status === "in_progress"
            ? "now"
            : "later",
    }));
  }
  const lastUser = messages.findLastIndex((message) => message.role === "user");
  return messages
    .slice(lastUser + 1)
    .filter((message) => message.role === "assistant")
    .flatMap((message) =>
      message.parts.flatMap((part, index) => {
        const row = toChainRow(part, index);
        if (!row || !row.tool || row.category === "narration") return [];
        return [
          {
            key: row.key,
            text: row.text,
            state:
              row.state === "running" ? ("now" as const) : ("done" as const),
          },
        ];
      }),
    );
}

/** The questions the user answered in this chat, newest first. */
export function decisionsOf(messages: UIMessage[]) {
  return messages
    .filter((message) => message.role === "user")
    .flatMap((message) => {
      const body = message.parts
        .flatMap((part) => (part.type === "text" ? [part.text] : []))
        .join("\n");
      return parseAnswersFrame(body) ?? [];
    })
    .toReversed();
}

const TYPE_BY_EXTENSION: Record<string, string> = {
  md: "Markdown",
  txt: "Text",
  doc: "Word",
  docx: "Word",
  pdf: "PDF",
  csv: "Sheet",
  xls: "Sheet",
  xlsx: "Sheet",
  ppt: "Slides",
  pptx: "Slides",
  json: "JSON",
  html: "Web page",
  png: "Image",
  jpg: "Image",
  jpeg: "Image",
  gif: "Image",
  webp: "Image",
  svg: "Image",
  mp4: "Video",
  mp3: "Audio",
  py: "Code",
  ts: "Code",
  tsx: "Code",
  js: "Code",
};

/** The kind of file, in a word, from its name. */
export function fileTypeLabel(name: string) {
  const extension = name.includes(".") ? name.split(".").pop() : "";
  return TYPE_BY_EXTENSION[(extension ?? "").toLowerCase()] ?? "File";
}

/** A text file whose first lines can stand as a preview. */
export function isPreviewable(name: string, mimeType: string) {
  return (
    mimeType.startsWith("text/") ||
    ["Markdown", "Text"].includes(fileTypeLabel(name))
  );
}

/** A title and the first lines of a text file, markdown marks taken off. */
export function previewOf(content: string, fallbackTitle: string) {
  const lines = content
    .split("\n")
    .map((line) => line.replace(/^[#>\-*\s]+/, "").trim())
    .filter(Boolean);
  const hasHeading = /^\s*#/.test(content);
  return {
    title: hasHeading && lines[0] ? lines[0] : fallbackTitle,
    lines: lines.slice(hasHeading ? 1 : 0, hasHeading ? 3 : 2),
  };
}
