import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { authHeaders } from "@/lib/api/client";
import { fileThumbnailURL } from "@/lib/api/fileThumbnails";
import { workspaceKeys, type WorkspaceFile } from "@/lib/api/workspace";

import { hasImageThumbnail } from "./listHelpers";

export function useFileThumbnail(file: WorkspaceFile) {
  // A failure may be a token that expired in the cached headers, so the
  // first one fetches fresh headers and tries once more; the second gives up.
  const [failures, setFailures] = useState(0);
  const wantsImage = hasImageThumbnail(file) && failures < 2;
  // One header read serves every row on screen.
  const headers = useQuery({
    queryKey: workspaceKeys.thumbnailHeaders,
    queryFn: authHeaders,
    enabled: wantsImage,
  });
  const isRefreshing = failures === 1 && headers.isFetching;

  function handleError() {
    setFailures((count) => count + 1);
    if (failures === 0) void headers.refetch();
  }

  return {
    image:
      wantsImage && headers.data && !isRefreshing
        ? { uri: fileThumbnailURL(file.id), headers: headers.data }
        : null,
    // A new key loads the image again with the new headers.
    imageKey: `${failures}-${headers.dataUpdatedAt}`,
    handleError,
  };
}
