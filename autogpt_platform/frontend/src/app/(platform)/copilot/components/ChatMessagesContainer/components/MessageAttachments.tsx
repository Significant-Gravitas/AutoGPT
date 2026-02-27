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
}

export function MessageAttachments({ files }: Props) {
  if (files.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-2">
      {files.map((file, i) => (
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
      ))}
    </div>
  );
}
