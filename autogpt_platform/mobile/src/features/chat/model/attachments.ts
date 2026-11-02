/**
 * The backend appends an `[Attached files]` block (`build_files_block`) to a
 * user message that carries files or folders. It is shown as attachment
 * chips rather than raw text.
 */
export interface MessageAttachment {
  kind: "file" | "folder";
  id: string;
  name: string;
  detail: string;
}

const FILE_HINT =
  "Use read_workspace_file with the file_id to access file contents.";
const FOLDER_HINT =
  "Use list_workspace_files with the folder_id to see what is in a folder.";
const HINTS = `(?:${escape(FILE_HINT)}|${escape(FOLDER_HINT)})`;
const ATTACHED_FILES_RE = new RegExp(
  `\\n?\\n?\\[Attached files\\]\\n([\\s\\S]*?)${HINTS}(?:\\n${HINTS})*`,
);
const FILE_LINE_RE = /^- (.+) \(([^,]+),\s*[\d.]+ KB\), file_id=([0-9a-f-]+)$/;
const FOLDER_LINE_RE =
  /^- (.+) \(folder, (\d+) file\(s\) directly inside\), folder_id=([0-9a-f-]+)$/;

export function splitAttachments(content: string): {
  text: string;
  attachments: MessageAttachment[];
} {
  const match = content.match(ATTACHED_FILES_RE);
  if (!match) return { text: content, attachments: [] };
  const attachments: MessageAttachment[] = [];
  for (const line of match[1].trim().split("\n")) {
    const trimmed = line.trim();
    const folder = trimmed.match(FOLDER_LINE_RE);
    if (folder) {
      attachments.push({
        kind: "folder",
        id: folder[3],
        name: folder[1],
        detail: `${folder[2]} files`,
      });
      continue;
    }
    const file = trimmed.match(FILE_LINE_RE);
    if (file)
      attachments.push({
        kind: "file",
        id: file[3],
        name: file[1],
        detail: file[2],
      });
  }
  return { text: content.replace(match[0], "").trim(), attachments };
}

function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
