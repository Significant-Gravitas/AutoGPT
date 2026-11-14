import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { authHeaders } from "@/lib/api/client";
import { fileThumbnailURL } from "@/lib/api/fileThumbnails";
import { workspaceKeys, type WorkspaceFile } from "@/lib/api/workspace";

import { hasImageThumbnail } from "./listHelpers";

export function useFileThumbnail(file: WorkspaceFile) {
  const [failed, setFailed] = useState(false);
  const wantsImage = hasImageThumbnail(file) && !failed;
  // Shared with the viewer: one header read serves every row on screen.
  const headers = useQuery({
    queryKey: workspaceKeys.imageHeaders,
    queryFn: authHeaders,
    enabled: wantsImage,
  });

  function handleError() {
    setFailed(true);
  }

  return {
    image:
      wantsImage && headers.data
        ? { uri: fileThumbnailURL(file.id), headers: headers.data }
        : null,
    handleError,
  };
}
