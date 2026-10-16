import type { FileUIPart } from "ai";
import {
  globalRegistry,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import type { OutputMetadata } from "@/components/contextual/OutputRenderers";
import {
  ContentCard,
  ContentCardHeader,
  ContentCardTitle,
  ContentCardSubtitle,
} from "../../ToolAccordion/AccordionContent";
import { ArtifactCard } from "../../ArtifactCard/ArtifactCard";
import { filePartToArtifactRef } from "../helpers";
import { Download04Icon, File02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  files: FileUIPart[];
  /** URL→file-ID pattern used by ``filePartToArtifactRef``.  Owner
   *  side defaults to the workspace-file URL shape; the public viewer
   *  passes a per-token pattern from ``lib/share/routes.ts``. */
  filePattern?: RegExp;
  /** Read-only mode — forwarded to ``ArtifactCard`` so clicks
   *  download instead of trying to open a panel that isn't mounted. */
  readOnly?: boolean;
}

function renderFileContent(file: FileUIPart): React.ReactNode | null {
  if (!file.url) return null;
  const metadata: OutputMetadata = {
    mimeType: file.mediaType,
    filename: file.filename,
    type: file.mediaType?.startsWith("image/")
      ? "image"
      : file.mediaType?.startsWith("video/")
        ? "video"
        : undefined,
  };
  const renderer = globalRegistry.getRenderer(file.url, metadata);
  if (!renderer) return null;
  return (
    <OutputItem value={file.url} metadata={metadata} renderer={renderer} />
  );
}

export function MessageAttachments({ files, filePattern, readOnly }: Props) {
  if (files.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-2">
      {files.map((file, i) => {
        const artifactRef = filePartToArtifactRef(file, "agent", filePattern);
        if (artifactRef) {
          return (
            <ArtifactCard
              key={`artifact-${artifactRef.id}-${i}`}
              artifact={artifactRef}
              readOnly={readOnly}
            />
          );
        }
        const rendered = renderFileContent(file);
        return rendered ? (
          <div
            key={`${file.filename}-${i}`}
            className="inline-block rounded-lg border border-neutral-200 bg-neutral-50 p-1.5"
          >
            {rendered}
            <div className="mt-1 flex items-center gap-1 px-0.5 text-xs text-neutral-500">
              <span className="truncate">{file.filename || "file"}</span>
              {file.url && (
                <a
                  href={file.url}
                  download
                  aria-label="Download file"
                  className="ml-auto shrink-0 opacity-50 hover:opacity-100"
                >
                  <Icon icon={Download04Icon} className="h-3.5 w-3.5" />
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
                    <Icon icon={Download04Icon} className="h-5 w-5" />
                  </a>
                ) : undefined
              }
            >
              <div className="flex items-center gap-2">
                <Icon
                  icon={File02Icon}
                  className="h-5 w-5 shrink-0 text-neutral-400"
                />
                <div className="min-w-0">
                  <ContentCardTitle>{file.filename || "file"}</ContentCardTitle>
                  <ContentCardSubtitle>
                    {file.mediaType || "file"}
                  </ContentCardSubtitle>
                </div>
              </div>
            </ContentCardHeader>
          </ContentCard>
        );
      })}
    </div>
  );
}
