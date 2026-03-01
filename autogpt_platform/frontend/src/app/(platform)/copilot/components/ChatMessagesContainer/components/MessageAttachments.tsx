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

interface Props {
  files: FileUIPart[];
  isUser?: boolean;
}

export function MessageAttachments({ files, isUser }: Props) {
  if (files.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-2">
      {files.map((file, i) =>
        isUser ? (
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
