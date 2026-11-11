import {
  BrainIcon,
  BubbleChatIcon,
  Clock01Icon,
  ComputerIcon,
  File01Icon,
  GlobalSearchIcon,
  Link01Icon,
  UserGroupIcon,
  WorkflowSquare01Icon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";
import {
  isToolUIPart,
  type DynamicToolUIPart,
  type ToolUIPart,
  type UIMessage,
} from "ai";

export type StepPart = ToolUIPart | DynamicToolUIPart;

export type StepKind =
  | "hidden"
  | "file"
  | "web"
  | "routine"
  | "memory"
  | "computer"
  | "connect"
  | "team"
  | "message"
  | "workflow"
  | "other";

export type VisibleStepKind = Exclude<StepKind, "hidden">;

export type StepState = "running" | "done" | "failed";

export interface Step {
  id: string;
  kind: VisibleStepKind;
  label: string;
  state: StepState;
}

export type Block =
  | { kind: "text"; key: string; text: string }
  | { kind: "thinking"; key: string }
  | { kind: "steps"; key: string; steps: Step[] };

export const stepIcons: Record<VisibleStepKind, IconSvgElement> = {
  file: File01Icon,
  web: GlobalSearchIcon,
  routine: Clock01Icon,
  memory: BrainIcon,
  computer: ComputerIcon,
  connect: Link01Icon,
  team: UserGroupIcon,
  message: BubbleChatIcon,
  workflow: WorkflowSquare01Icon,
  other: Wrench01Icon,
};

const WEB_TOOLS = ["web_search", "web_fetch", "search_docs", "get_doc_page"];

function hasAny(name: string, fragments: string[]) {
  return fragments.some((fragment) => name.includes(fragment));
}

/** Internal tool name: never render it or put it in an accessibility label. */
export function stepToolName(part: StepPart) {
  return part.type === "dynamic-tool"
    ? part.toolName
    : part.type.slice("tool-".length);
}

export function stepKind(toolName: string): StepKind {
  const name = toolName.toLowerCase();
  if (name === "todowrite") return "hidden";
  if (hasAny(name, ["workspace_file", "folder"])) return "file";
  if (WEB_TOOLS.includes(name) || name.startsWith("browser_")) return "web";
  if (hasAny(name, ["schedule", "routine", "followup"])) return "routine";
  if (name.startsWith("memory_") || name === "add_understanding") {
    return "memory";
  }
  if (name === "bash_exec" || name === "start_desktop") return "computer";
  if (name === "connect_integration" || name.includes("credential")) {
    return "connect";
  }
  if (hasAny(name, ["expert", "teammate"]) || name === "list_team") {
    return "team";
  }
  if (hasAny(name, ["chat_platform", "session"]) || name === "ask_question") {
    return "message";
  }
  if (hasAny(name, ["agent", "capability", "preset", "trigger", "skill"])) {
    return "workflow";
  }
  return "other";
}

export function stepLabel(part: StepPart) {
  const title =
    typeof part.title === "string" && part.title.trim()
      ? part.title
      : "Working on it";
  return part.state === "output-error" ? `${title} · didn't finish` : title;
}

export function stepState(part: StepPart): StepState {
  if (part.state === "input-streaming" || part.state === "input-available") {
    return "running";
  }
  if (part.state === "output-error") return "failed";
  return "done";
}

/**
 * Splits a message's parts into render blocks. Tool parts separated only by
 * parts that render nothing (step-start, finished reasoning, empty text)
 * share one steps block.
 */
export function groupParts(parts: UIMessage["parts"]): Block[] {
  const blocks: Block[] = [];
  let run: Step[] | null = null;

  function closeRun() {
    if (run && run.length > 0) {
      blocks.push({ kind: "steps", key: run[0].id, steps: run });
    }
    run = null;
  }

  parts.forEach((part, index) => {
    if (isToolUIPart(part)) {
      const kind = stepKind(stepToolName(part));
      run ??= [];
      if (kind === "hidden") return;
      run.push({
        id: part.toolCallId,
        kind,
        label: stepLabel(part),
        state: stepState(part),
      });
      return;
    }
    if (part.type === "text" && part.text.trim()) {
      closeRun();
      blocks.push({ kind: "text", key: `text-${index}`, text: part.text });
      return;
    }
    if (part.type === "reasoning" && part.state === "streaming") {
      closeRun();
      blocks.push({ kind: "thinking", key: `thinking-${index}` });
    }
  });
  closeRun();
  return blocks;
}
