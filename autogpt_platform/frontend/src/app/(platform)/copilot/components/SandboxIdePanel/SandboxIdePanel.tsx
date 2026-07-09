"use client";

import { FilesTab } from "./components/FilesTab/FilesTab";

interface Props {
  sessionId: string;
}

export function SandboxIdePanel({ sessionId }: Props) {
  // The IDE (files + terminal) also hosts artifact previews inside its editor
  // pane — see FilesTab.
  return (
    <div className="flex h-full w-full flex-col font-sans">
      <FilesTab sessionId={sessionId} />
    </div>
  );
}
