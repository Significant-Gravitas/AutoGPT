import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { authHeaders } from "@/lib/api/client";
import {
  downloadFileText,
  fileDownloadURL,
  isPreviewUnavailable,
  workspaceKeys,
  type WorkspaceFile,
} from "@/lib/api/workspace";

import { capText, fileKind } from "./helpers";

// File content can be large; drop it soon after the viewer closes.
const CONTENT_GC_TIME = 30_000;

export function useFileViewer(file: WorkspaceFile) {
  const kind = fileKind(file);
  // Bumped on retry: a new key remounts the image so it loads again.
  const [imageAttempt, setImageAttempt] = useState(0);
  const [imageFailed, setImageFailed] = useState(false);
  const text = useQuery({
    queryKey: workspaceKeys.fileText(file.id),
    queryFn: () => downloadFileText(file),
    enabled: kind === "markdown" || kind === "text",
    gcTime: CONTENT_GC_TIME,
    select: capText,
  });
  // The token can be re-minted between openings, so the headers are read
  // fresh for every image rather than kept.
  const headers = useQuery({
    queryKey: workspaceKeys.imageHeaders,
    queryFn: authHeaders,
    enabled: kind === "image",
    gcTime: 0,
  });
  const active = kind === "image" ? headers : text;

  function retry() {
    if (kind === "image") {
      setImageFailed(false);
      setImageAttempt((attempt) => attempt + 1);
    }
    void active.refetch();
  }

  function handleImageError() {
    setImageFailed(true);
  }

  return {
    kind,
    text: text.data?.text ?? null,
    cappedNotice: text.data?.cappedNotice ?? null,
    isTooLarge: isPreviewUnavailable(text.error),
    image: headers.data
      ? { uri: fileDownloadURL(file.id), headers: headers.data }
      : null,
    isLoading: kind !== "other" && active.isPending,
    imageAttempt,
    isError: imageFailed || (kind !== "other" && active.isError),
    retry,
    handleImageError,
  };
}
