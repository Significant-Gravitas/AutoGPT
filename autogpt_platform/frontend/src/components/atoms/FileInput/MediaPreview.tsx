import { parseWorkspaceURI } from "@/lib/workspace-uri";

type MediaPreviewData =
  | { type: "audio"; url: string }
  | { type: "video"; url: string; mimeType: string }
  | { type: "image"; url: string }
  | null;

export function getMediaPreview(
  value: string | undefined,
  contentType: string | undefined,
): MediaPreviewData {
  if (!value) return null;
  const mimeType = contentType || parseWorkspaceURI(value)?.mimeType || null;
  if (!mimeType) return null;

  const wsURI = parseWorkspaceURI(value);
  const previewURL = wsURI
    ? `/api/proxy/api/workspace/files/${wsURI.fileID}/download`
    : value.startsWith("data:")
      ? value
      : null;
  if (!previewURL) return null;

  if (mimeType.startsWith("audio/")) return { type: "audio", url: previewURL };
  if (mimeType.startsWith("video/"))
    return { type: "video", url: previewURL, mimeType };
  if (mimeType.startsWith("image/")) return { type: "image", url: previewURL };
  return null;
}

export function MediaPreviewRenderer({
  preview,
}: {
  preview: MediaPreviewData;
}) {
  if (!preview) return null;
  switch (preview.type) {
    case "audio":
      return (
        <audio controls preload="metadata" className="w-full" src={preview.url}>
          Your browser does not support the audio element.
        </audio>
      );
    case "video":
      return (
        <video
          controls
          preload="metadata"
          className="h-auto max-w-full rounded-md"
        >
          <source src={preview.url} type={preview.mimeType} />
          Your browser does not support the video tag.
        </video>
      );
    case "image":
      return (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={preview.url}
          alt="Uploaded file"
          className="h-auto max-w-full rounded-md"
          loading="lazy"
        />
      );
  }
}
