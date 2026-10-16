import {
  getFileTypeIcon,
  getFileTypeLabel,
} from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { classifyArtifact } from "../ArtifactPanel/helpers";

/** The `mediaType` an attachment card takes for a workspace folder. */
export const FOLDER_MEDIA_TYPE = "folder";

export function formatSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function isImageType(mediaType: string): boolean {
  return mediaType.toLowerCase().startsWith("image/");
}

export function attachmentIcon(mediaType: string, name: string) {
  if (mediaType === FOLDER_MEDIA_TYPE) return Folder01Icon;
  return getFileTypeIcon(mediaType, name);
}

/** "PDF · 1.2 MB", or just "PDF" while the size is unknown. */
export function attachmentDetail(
  mediaType: string,
  name: string,
  sizeBytes?: number,
): string {
  if (mediaType === FOLDER_MEDIA_TYPE) return "Folder";
  const label = typeLabel(mediaType, name, sizeBytes);
  const size = formatSize(sizeBytes);
  return size ? `${label} · ${size}` : label;
}

/** The artifact panel's short label ("PDF", "Code"), except where it can
 *  only say "File": office documents and oversized files still get named. */
function typeLabel(mediaType: string, name: string, sizeBytes?: number) {
  const { label } = classifyArtifact(mediaType, name, sizeBytes);
  if (label !== "File") return label;
  const listed = getFileTypeLabel(mediaType, name);
  return listed === "Generated file" ? label : listed;
}
