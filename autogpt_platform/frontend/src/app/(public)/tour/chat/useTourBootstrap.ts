"use client";

import {
  getListWorkspaceFilesQueryKey,
  type listWorkspaceFilesResponse,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { useMountEffect } from "@/hooks/useMountEffect";
import { useQueryClient } from "@tanstack/react-query";
import { tourSessionIds } from "./script/tourChats";

const emptyFiles: ListFilesResponse = { files: [], offset: 0, has_more: false };

const emptyWorkspaceFilesResponse: listWorkspaceFilesResponse = {
  status: 200,
  data: emptyFiles,
  headers: new Headers(),
};

export function useTourBootstrap() {
  const queryClient = useQueryClient();

  useMountEffect(() => {
    for (const sessionId of tourSessionIds) {
      const key = getListWorkspaceFilesQueryKey({ session_id: sessionId });
      queryClient.setQueryData(key, emptyWorkspaceFilesResponse);
      queryClient.setQueryDefaults(key, {
        staleTime: Infinity,
        gcTime: Infinity,
        retry: false,
        refetchOnMount: false,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
      });
    }
  });
}
