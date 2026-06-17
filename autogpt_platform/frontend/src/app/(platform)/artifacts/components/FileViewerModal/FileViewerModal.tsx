"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { ArtifactContent } from "@/app/(platform)/copilot/components/ArtifactPanel/components/ArtifactContent";
import { classifyArtifact } from "@/app/(platform)/copilot/components/ArtifactPanel/helpers";
import type { ArtifactRef } from "@/app/(platform)/copilot/store";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CodeIcon, DownloadSimpleIcon, EyeIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { downloadFileBlob, getFileDownloadUrl } from "../ArtifactsList/helpers";

interface Props {
  file: WorkspaceFileItem | null;
  onClose: () => void;
}

const VIEWER_STYLING = {
  width: "90vw",
  maxWidth: "1100px",
  height: "85vh",
  padding: "1rem",
};

export function FileViewerModal({ file, onClose }: Props) {
  // Parent conditionally renders this and keys it by file id, so each open
  // mounts a fresh instance and per-file view state starts at preview.
  const [isSourceView, setIsSourceView] = useState(false);

  if (!file) return null;

  const classification = classifyArtifact(
    file.mime_type ?? null,
    file.name,
    file.size_bytes,
  );
  const artifact = toArtifactRef(file);

  return (
    <Dialog
      controlled={{
        isOpen: true,
        set: (open) => {
          if (!open) onClose();
        },
      }}
      styling={VIEWER_STYLING}
      title={
        <Header
          name={file.name}
          fileId={file.id}
          showSourceToggle={classification.hasSourceToggle}
          isSourceView={isSourceView}
          onToggleSource={() => setIsSourceView((v) => !v)}
        />
      }
    >
      <Dialog.Content>
        <div className="flex h-full min-h-0 flex-col" data-testid="file-viewer">
          {classification.openable ? (
            <ArtifactContent
              artifact={artifact}
              isSourceView={isSourceView}
              classification={classification}
            />
          ) : (
            <DownloadOnly
              name={file.name}
              downloadUrl={getFileDownloadUrl(file.id)}
            />
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

interface HeaderProps {
  name: string;
  fileId: string;
  showSourceToggle: boolean;
  isSourceView: boolean;
  onToggleSource: () => void;
}

function Header({
  name,
  fileId,
  showSourceToggle,
  isSourceView,
  onToggleSource,
}: HeaderProps) {
  const { toast } = useToast();
  const [isDownloading, setIsDownloading] = useState(false);

  async function handleDownload() {
    if (isDownloading) return;
    setIsDownloading(true);
    try {
      await downloadFileBlob(fileId, name);
    } catch (error) {
      toast({
        title: "Failed to download file",
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }

  return (
    // pr-12 leaves room for the Dialog's absolute close button.
    <div className="flex w-full flex-col gap-2 pr-12">
      <span className="min-w-0 truncate text-base font-medium text-zinc-900">
        {name}
      </span>
      <div className="flex items-center gap-2">
        {showSourceToggle ? (
          <Button
            variant="secondary"
            size="small"
            onClick={onToggleSource}
            leftIcon={
              isSourceView ? <EyeIcon size={14} /> : <CodeIcon size={14} />
            }
          >
            {isSourceView ? "Preview" : "Source"}
          </Button>
        ) : null}
        <Button
          variant="secondary"
          size="small"
          onClick={handleDownload}
          loading={isDownloading}
          leftIcon={<DownloadSimpleIcon size={14} />}
          data-testid="file-viewer-download"
        >
          {isDownloading ? "Downloading…" : "Download"}
        </Button>
      </div>
    </div>
  );
}

function DownloadOnly({
  name,
  downloadUrl,
}: {
  name: string;
  downloadUrl: string;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
      <p className="text-sm text-zinc-500">
        This file type can&apos;t be previewed.
      </p>
      <a
        href={downloadUrl}
        download={name}
        className="inline-flex items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50"
      >
        <DownloadSimpleIcon size={16} />
        Download
      </a>
    </div>
  );
}

function toArtifactRef(file: WorkspaceFileItem): ArtifactRef {
  return {
    id: file.id,
    title: file.name,
    mimeType: file.mime_type ?? null,
    sourceUrl: getFileDownloadUrl(file.id),
    origin: file.origin === "uploaded" ? "user-upload" : "agent",
    sizeBytes: file.size_bytes,
  };
}
