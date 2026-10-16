import type { PendingUploadStatus } from "../../../copilotStreamStore";

interface Props {
  status?: PendingUploadStatus;
}

/** Names the upload state for screen readers; the card shows it with a
 *  spinner, the file's own icon, or the failed styling. */
export function UploadStatusText({ status }: Props) {
  if (status === "uploading") return <span className="sr-only">Uploading</span>;
  if (status === "uploaded") return <span className="sr-only">Uploaded</span>;
  return null;
}
