import { UIDataTypes, UIMessage, UITools } from "ai";
import type { ArtifactRef } from "../../store";
import { classifyArtifactRef } from "../ArtifactPanel/helpers";
import { defaultWorkspaceFileUrl } from "../ChatMessagesContainer/helpers";
import { asObject, str } from "../ToolChain/resultHelpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

export interface ArtifactOverwrite {
  replacedFileId: string;
  next: ArtifactRef;
}

const WRITE_TOOL_PART_TYPE = "tool-write_workspace_file";

function partOverwrite(
  part: Message["parts"][number],
): ArtifactOverwrite | null {
  if (part.type !== WRITE_TOOL_PART_TYPE || !("output" in part)) return null;
  const output = asObject(part.output);
  if (!output) return null;
  const replacedFileId = str(output, "replaced_file_id");
  const fileId = str(output, "file_id");
  if (!replacedFileId || !fileId) return null;
  return {
    replacedFileId,
    next: {
      id: fileId,
      title: str(output, "name") ?? `File ${fileId.slice(0, 8)}`,
      mimeType: str(output, "mime_type"),
      sourceUrl: defaultWorkspaceFileUrl(fileId),
      origin: "agent",
      sizeBytes:
        typeof output.size_bytes === "number" ? output.size_bytes : undefined,
    },
  };
}

/**
 * The files the latest assistant turn overwrote, each paired with its last
 * version in that turn. Reads that turn only, since this runs on every
 * streaming tick.
 *
 * A last version too big to preview is left out: cards download those instead
 * of opening them, and the panel would fetch and render the whole file.
 */
export function getLatestTurnOverwrites(
  messages: Message[],
): ArtifactOverwrite[] {
  let end = messages.length;
  while (end > 0 && messages[end - 1].role !== "assistant") end--;
  let start = end;
  while (start > 0 && messages[start - 1].role === "assistant") start--;

  const writes: ArtifactOverwrite[] = [];
  for (const message of messages.slice(start, end)) {
    for (const part of message.parts) {
      const overwrite = partOverwrite(part);
      if (overwrite) writes.push(overwrite);
    }
  }

  // Newest first, so a file's successor already points at the end of the chain.
  const lastVersions = new Map<string, ArtifactRef>();
  for (const { replacedFileId, next } of writes.reverse()) {
    lastVersions.set(replacedFileId, lastVersions.get(next.id) ?? next);
  }
  return Array.from(lastVersions, ([replacedFileId, next]) => ({
    replacedFileId,
    next,
  })).filter(({ next }) => classifyArtifactRef(next).openable);
}
