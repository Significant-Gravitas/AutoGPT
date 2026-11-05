import { UIDataTypes, UIMessage, UITools } from "ai";
import type { ArtifactRef } from "../../store";
import { defaultWorkspaceFileUrl } from "../ChatMessagesContainer/helpers";
import { asObject, str } from "../ToolChain/resultHelpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

export interface ArtifactOverwrite {
  replacedFileId: string;
  next: ArtifactRef;
}

const WRITE_TOOL_PART_TYPE = "tool-write_workspace_file";

function partOverwrite(part: Message["parts"][number]) {
  if (part.type !== WRITE_TOOL_PART_TYPE || !("output" in part)) return null;
  const output = asObject(part.output);
  if (!output) return null;
  const replacedFileId = str(output, "replaced_file_id");
  const fileId = str(output, "file_id");
  if (!replacedFileId || !fileId) return null;
  const next: ArtifactRef = {
    id: fileId,
    title: str(output, "name") ?? `File ${fileId.slice(0, 8)}`,
    mimeType: str(output, "mime_type"),
    sourceUrl: defaultWorkspaceFileUrl(fileId),
    origin: "agent",
    sizeBytes:
      typeof output.size_bytes === "number" ? output.size_bytes : undefined,
  };
  return { replacedFileId, next };
}

/**
 * The files the latest assistant turn overwrote, oldest first, so a file
 * rewritten twice in one turn is followed through to its last version. Reads
 * that turn only, since this runs on every streaming tick.
 */
export function getLatestTurnOverwrites(
  messages: Message[],
): ArtifactOverwrite[] {
  let end = messages.length;
  while (end > 0 && messages[end - 1].role !== "assistant") end--;
  let start = end;
  while (start > 0 && messages[start - 1].role === "assistant") start--;

  const overwrites: ArtifactOverwrite[] = [];
  for (const message of messages.slice(start, end)) {
    for (const part of message.parts) {
      const overwrite = partOverwrite(part);
      if (overwrite) overwrites.push(overwrite);
    }
  }
  return overwrites;
}
