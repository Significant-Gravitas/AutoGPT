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
import { isPreviewableImage } from "../../ArtifactPanel/helpers";
import type { ArtifactRef } from "../../../store";
import { filePartToArtifactRef } from "../helpers";
import { MessageImageThumbnail } from "./MessageImageThumbnail";
import {
  Download04Icon,
  File02Icon,
  Folder01Icon,
  LinkSquare01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import Link from "next/link";
import { folderSummary } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import type { WorkspaceFolderPartData } from "../../../helpers/workspaceAttachments";

interface Props {
  files: FileUIPart[];
  /** Folders attached to the message — a pointer for the model, never
   *  expanded into files, so they never take the ArtifactCard path. */
  folders?: WorkspaceFolderPartData[];
  isUser?: boolean;
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

export function MessageAttachments({
  files,
  folders = [],
  isUser,
  filePattern,
  readOnly,
}: Props) {
  if (files.length === 0 && folders.length === 0) return null;

  const { images, otherFiles } = splitUserImages(files, isUser, filePattern);

  return (
    <div className="mt-2 flex flex-col gap-2">
      {images.length > 0 && (
        <div className="flex flex-wrap justify-end gap-2">
          {images.map((image, i) => (
            <MessageImageThumbnail
              key={`image-${image.id}-${i}`}
              artifact={image}
              readOnly={readOnly}
            />
          ))}
        </div>
      )}
      {folders.map((folder) => (
        <FolderAttachmentCard
          key={folder.id}
          folder={folder}
          readOnly={readOnly}
        />
      ))}
      {otherFiles.map((file, i) => {
        const artifactRef = filePartToArtifactRef(
          file,
          isUser ? "user-upload" : "agent",
          filePattern,
        );
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
            className={`inline-block rounded-lg border p-1.5 ${
              isUser
                ? // TODO(tokens): no exact role for border-purple-300, bg-purple-50
                  "border-purple-300 bg-purple-50"
                : // TODO(tokens): no exact role for border-neutral-200
                  "border-neutral-200 bg-background"
            }`}
          >
            {rendered}
            <div
              className={`mt-1 flex items-center gap-1 px-0.5 text-xs ${
                // TODO(tokens): no exact role for text-neutral-500
                isUser ? "text-foreground-secondary" : "text-neutral-500"
              }`}
            >
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
        ) : isUser ? (
          <div
            key={`${file.filename}-${i}`}
            // TODO(tokens): no exact role for border-purple-300, bg-purple-100
            className="min-w-0 rounded-lg border border-purple-300 bg-purple-100 p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex min-w-0 items-center gap-2">
                <Icon
                  icon={File02Icon}
                  // TODO(tokens): no exact role for text-neutral-400
                  className="h-5 w-5 shrink-0 text-neutral-400"
                />
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-foreground-emphasis">
                    {file.filename || "file"}
                  </p>
                  <p className="mt-0.5 truncate font-mono text-xs text-foreground-emphasis">
                    {file.mediaType || "file"}
                  </p>
                </div>
              </div>
              {file.url && (
                <a
                  href={file.url}
                  download
                  aria-label="Download file"
                  // TODO(tokens): no exact role for text-purple-400; no fitting role for hover:text-purple-600 #6C2EDF; #6C2EDF is brand-hover (a fill)
                  className="shrink-0 text-purple-400 hover:text-purple-600"
                >
                  <Icon icon={Download04Icon} className="h-5 w-5" />
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
                    // TODO(tokens): no exact role for text-neutral-400, hover:text-neutral-600
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
                  // TODO(tokens): no exact role for text-neutral-400
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

/** Images the user sent show as pictures; everything else keeps its card. */
function splitUserImages(
  files: FileUIPart[],
  isUser: boolean | undefined,
  filePattern: RegExp | undefined,
) {
  const images: ArtifactRef[] = [];
  const otherFiles: FileUIPart[] = [];
  for (const file of files) {
    const ref =
      isUser && isPreviewableImage(file.mediaType)
        ? filePartToArtifactRef(file, "user-upload", filePattern)
        : null;
    if (ref) images.push(ref);
    else otherFiles.push(file);
  }
  return { images, otherFiles };
}

function FolderAttachmentCard({
  folder,
  readOnly,
}: {
  folder: WorkspaceFolderPartData;
  readOnly?: boolean;
}) {
  return (
    <div
      // TODO(tokens): no exact role for border-purple-300, bg-purple-100
      className="flex min-w-0 items-center gap-3 rounded-lg border border-purple-300 bg-purple-100 p-3"
    >
      <Icon
        icon={Folder01Icon}
        size={20}
        className="shrink-0 text-foreground-body"
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium text-foreground-emphasis">
          {folder.name}
        </span>
        <span className="text-xs text-foreground-secondary">
          {/* The part counts direct files only, so zero is not "Empty". */}
          {folder.fileCount > 0
            ? `Folder · ${folderSummary(folder.fileCount, 0)}`
            : "Folder"}
        </span>
      </div>
      {/* The share viewer has no Files page to send anyone to. */}
      {readOnly ? null : (
        <Link
          href={`/artifacts?folder=${encodeURIComponent(folder.id)}`}
          aria-label={`Open folder ${folder.name} in Files`}
          // TODO(tokens): no exact role for hover:bg-purple-200
          className="shrink-0 rounded-md p-1 text-foreground-tertiary transition-colors hover:bg-purple-200 hover:text-foreground-emphasis"
        >
          <Icon icon={LinkSquare01Icon} size={16} />
        </Link>
      )}
    </div>
  );
}
