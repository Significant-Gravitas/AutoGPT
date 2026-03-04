import {
  FileText as FileTextIcon,
  DownloadSimple as DownloadIcon,
} from "@phosphor-icons/react";
import type { FileUIPart } from "ai";
import {
  ContentCard,
  ContentCardHeader,
  ContentCardTitle,
  ContentCardSubtitle,
} from "../../ToolAccordion/AccordionContent";
import { WorkspaceMediaImage } from "./MessagePartRenderer";

interface Props {
  files: FileUIPart[];
  isUser?: boolean;
}

function isImageMime(mime?: string): boolean {
  return !!mime && mime.startsWith("image/");
}

function ImageAttachment({
  file,
  isUser,
}: {
  file: FileUIPart;
  isUser?: boolean;
}) {
  const borderClass = isUser
    ? "border-purple-300 bg-purple-50"
    : "border-neutral-200 bg-neutral-50";
  const nameClass = isUser ? "text-zinc-600" : "text-neutral-500";
  return (
    <div className={`inline-block rounded-lg border ${borderClass} p-1.5`}>
      <WorkspaceMediaImage
        src={file.url}
        alt={file.filename || "image"}
        className="max-h-48 rounded"
      />
      <div
        className={`mt-1 flex items-center gap-1 px-0.5 text-xs ${nameClass}`}
      >
        <span className="truncate">{file.filename || "image"}</span>
        {file.url && (
          <a
            href={file.url}
            download
            aria-label="Download file"
            className="ml-auto shrink-0 opacity-50 hover:opacity-100"
          >
            <DownloadIcon className="h-3.5 w-3.5" />
          </a>
        )}
      </div>
    </div>
  );
}

export function MessageAttachments({ files, isUser }: Props) {
  if (files.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-2">
      {files.map((file, i) =>
        isImageMime(file.mediaType) && file.url ? (
          <ImageAttachment
            key={`${file.filename}-${i}`}
            file={file}
            isUser={isUser}
          />
        ) : isUser ? (
          <div
            key={`${file.filename}-${i}`}
            className="min-w-0 rounded-lg border border-purple-300 bg-purple-100 p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex min-w-0 items-center gap-2">
                <FileTextIcon className="h-5 w-5 shrink-0 text-neutral-400" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-zinc-800">
                    {file.filename || "file"}
                  </p>
                  <p className="mt-0.5 truncate font-mono text-xs text-zinc-800">
                    {file.mediaType || "file"}
                  </p>
                </div>
              </div>
              {file.url && (
                <a
                  href={file.url}
                  download
                  aria-label="Download file"
                  className="shrink-0 text-purple-400 hover:text-purple-600"
                >
                  <DownloadIcon className="h-5 w-5" />
                </a>
              )}
            </div>
          </div>
        ) : (
          <ContentCard key={`${file.filename}-${i}`}>
            <ContentCardHeader
              action={
                file.url ? (
                  <a
                    href={file.url}
                    download
                    aria-label="Download file"
                    className="shrink-0 text-neutral-400 hover:text-neutral-600"
                  >
                    <DownloadIcon className="h-5 w-5" />
                  </a>
                ) : undefined
              }
            >
              <div className="flex items-center gap-2">
                <FileTextIcon className="h-5 w-5 shrink-0 text-neutral-400" />
                <div className="min-w-0">
                  <ContentCardTitle>{file.filename || "file"}</ContentCardTitle>
                  <ContentCardSubtitle>
                    {file.mediaType || "file"}
                  </ContentCardSubtitle>
                </div>
              </div>
            </ContentCardHeader>
          </ContentCard>
        ),
      )}
    </div>
  );
}
