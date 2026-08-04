import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import type { MessagePart } from "../components/ChatMessagesContainer/helpers";

export type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

export type SampleEvent =
  | { delay: number; kind: "user"; id: string; text: string }
  // Playback pauses here until the user submits an answer through an
  // interactive card (e.g. ask_question); the submitted text becomes the
  // next user message.
  | { delay: number; kind: "await-user" }
  // Mirrors the backend's `data-status` events: transient copy for the
  // thinking indicator during a silent gap. Any content event clears it.
  | { delay: number; kind: "status"; message: string }
  | { delay: number; kind: "assistant-start"; id: string }
  | { delay: number; kind: "text-start"; messageId: string }
  | { delay: number; kind: "text-delta"; messageId: string; delta: string }
  | { delay: number; kind: "reasoning-start"; messageId: string }
  | { delay: number; kind: "reasoning-delta"; messageId: string; delta: string }
  | { delay: number; kind: "reasoning-done"; messageId: string }
  | {
      delay: number;
      kind: "tool-start";
      messageId: string;
      toolCallId: string;
      tool: string;
      input: unknown;
    }
  | {
      delay: number;
      kind: "tool-output";
      messageId: string;
      toolCallId: string;
      output: unknown;
    }
  | {
      delay: number;
      kind: "tool-error";
      messageId: string;
      toolCallId: string;
      errorText: string;
    };

function updateMessage(
  messages: ChatMessage[],
  messageId: string,
  updateParts: (parts: MessagePart[]) => MessagePart[],
): ChatMessage[] {
  return messages.map((message) =>
    message.id === messageId
      ? { ...message, parts: updateParts(message.parts as MessagePart[]) }
      : message,
  );
}

function appendToLastPart(
  parts: MessagePart[],
  type: "text" | "reasoning",
  delta: string,
  done?: boolean,
): MessagePart[] {
  const lastIndex = parts.findLastIndex((p) => p.type === type);
  if (lastIndex === -1) return parts;
  return parts.map((part, i) => {
    if (i !== lastIndex) return part;
    const prev = part as { text: string };
    return {
      ...part,
      text: prev.text + delta,
      ...(type === "reasoning" && done !== undefined
        ? { state: done ? "done" : "streaming" }
        : {}),
    } as MessagePart;
  });
}

function updateToolPart(
  parts: MessagePart[],
  toolCallId: string,
  patch: Partial<ToolUIPart>,
): MessagePart[] {
  return parts.map((part) =>
    part.type.startsWith("tool-") &&
    (part as ToolUIPart).toolCallId === toolCallId
      ? ({ ...part, ...patch } as MessagePart)
      : part,
  );
}

export function applyEvent(
  messages: ChatMessage[],
  event: SampleEvent,
): ChatMessage[] {
  switch (event.kind) {
    case "await-user":
    case "status":
      return messages;
    case "user":
      return [
        ...messages,
        {
          id: event.id,
          role: "user",
          parts: [{ type: "text", text: event.text }],
        },
      ];
    case "assistant-start":
      return [...messages, { id: event.id, role: "assistant", parts: [] }];
    case "text-start":
      return updateMessage(messages, event.messageId, (parts) => [
        ...parts,
        { type: "text", text: "" },
      ]);
    case "text-delta":
      return updateMessage(messages, event.messageId, (parts) =>
        appendToLastPart(parts, "text", event.delta),
      );
    case "reasoning-start":
      return updateMessage(messages, event.messageId, (parts) => [
        ...parts,
        { type: "reasoning", text: "", state: "streaming" } as MessagePart,
      ]);
    case "reasoning-delta":
      return updateMessage(messages, event.messageId, (parts) =>
        appendToLastPart(parts, "reasoning", event.delta, false),
      );
    case "reasoning-done":
      return updateMessage(messages, event.messageId, (parts) =>
        appendToLastPart(parts, "reasoning", "", true),
      );
    case "tool-start":
      return updateMessage(messages, event.messageId, (parts) => [
        ...parts,
        {
          type: `tool-${event.tool}`,
          toolCallId: event.toolCallId,
          state: "input-available",
          input: event.input,
        } as MessagePart,
      ]);
    case "tool-output":
      return updateMessage(messages, event.messageId, (parts) =>
        updateToolPart(parts, event.toolCallId, {
          state: "output-available",
          output: event.output,
        } as Partial<ToolUIPart>),
      );
    case "tool-error":
      return updateMessage(messages, event.messageId, (parts) =>
        updateToolPart(parts, event.toolCallId, {
          state: "output-error",
          errorText: event.errorText,
        } as Partial<ToolUIPart>),
      );
  }
}
