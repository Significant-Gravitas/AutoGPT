import { useState } from "react";
import type { Attachment } from "../../helpers/workspaceAttachments";

export function useChatInputDraft() {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  return { value, setValue, attachments, setAttachments };
}
