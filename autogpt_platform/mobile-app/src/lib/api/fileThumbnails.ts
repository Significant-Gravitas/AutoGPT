import { apiUrl } from "./client";

// The width the web asks the preview route for (FileThumbnail on /artifacts).
const THUMBNAIL_WIDTH = 96;

/** A small WebP of an image, sized by the preview route. */
export function fileThumbnailURL(fileID: string) {
  return apiUrl(`/api/workspace/files/${encodeURIComponent(fileID)}/preview`, {
    w: THUMBNAIL_WIDTH,
  });
}
