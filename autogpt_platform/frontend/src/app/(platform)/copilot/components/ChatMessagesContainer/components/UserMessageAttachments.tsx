"use client";

import { folderSummary } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import { toast } from "@/components/molecules/Toast/use-toast";
import type { FileUIPart } from "ai";
import {
  filePartSize,
  type WorkspaceFolderPartData,
} from "../../../helpers/workspaceAttachments";
import { useCopilotUIStore } from "../../../store";
import { AttachmentCard } from "../../AttachmentCard/AttachmentCard";
import { FOLDER_MEDIA_TYPE } from "../../AttachmentCard/helpers";
import { downloadArtifact } from "../../ArtifactPanel/downloadArtifact";
import { classifyArtifact } from "../../ArtifactPanel/helpers";
import { filePartToArtifactRef } from "../helpers";
import { UserAttachmentRow } from "./UserAttachmentRow";

interface Props {
  files: FileUIPart[];
  folders: WorkspaceFolderPartData[];
  /** URL→file-ID pattern for `filePartToArtifactRef`; the public viewer
   *  passes its per-token shape. */
  filePattern?: RegExp;
  /** The share viewer has no Files page, so a folder links nowhere. */
  readOnly?: boolean;
}

/** What the user attached to a sent message, above its bubble. */
export function UserMessageAttachments({
  files,
  folders,
  filePattern,
  readOnly,
}: Props) {
  if (files.length === 0 && folders.length === 0) return null;

  return (
    <UserAttachmentRow>
      {folders.map((folder) => (
        <AttachmentCard
          key={folder.id}
          size="message"
          name={folder.name}
          mediaType={FOLDER_MEDIA_TYPE}
          // The part counts direct files only, so zero is not "Empty".
          detail={
            folder.fileCount > 0
              ? `Folder · ${folderSummary(folder.fileCount, 0)}`
              : "Folder"
          }
          href={
            readOnly
              ? undefined
              : `/artifacts?folder=${encodeURIComponent(folder.id)}`
          }
        />
      ))}
      {files.map((file, i) => (
        <SentFileCard
          key={`${file.url}-${i}`}
          file={file}
          filePattern={filePattern}
        />
      ))}
    </UserAttachmentRow>
  );
}

interface SentFileProps {
  file: FileUIPart;
  filePattern?: RegExp;
}

function SentFileCard({ file, filePattern }: SentFileProps) {
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const artifact = filePartToArtifactRef(file, "user-upload", filePattern);
  const name = file.filename || "file";
  const mediaType = file.mediaType || "application/octet-stream";
  const sizeBytes = filePartSize(file);

  function handleOpen() {
    if (!artifact) return;
    // Same split as the assistant's ArtifactCard: what the panel can show
    // opens there, anything else downloads.
    if (
      classifyArtifact(artifact.mimeType, artifact.title, sizeBytes).openable
    ) {
      openArtifact({ ...artifact, sizeBytes });
      return;
    }
    downloadArtifact(artifact).catch(() => {
      toast({
        title: "Download failed",
        description: "Couldn't fetch the file.",
        variant: "destructive",
      });
    });
  }

  return (
    <AttachmentCard
      size="message"
      name={name}
      mediaType={mediaType}
      sizeBytes={sizeBytes}
      previewUrl={file.url}
      onOpen={artifact ? handleOpen : undefined}
    />
  );
}
