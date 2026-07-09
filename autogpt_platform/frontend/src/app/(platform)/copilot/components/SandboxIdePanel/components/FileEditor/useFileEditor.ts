import {
  getGetV2GetSandboxChangesQueryKey,
  getGetV2GetSandboxFileQueryKey,
  useGetV2GetSandboxFile,
  usePutV2WriteSandboxFile,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SandboxFileResponse } from "@/app/api/__generated__/models/sandboxFileResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

export function useFileEditor(sessionId: string, path: string) {
  const queryClient = useQueryClient();
  const [value, setValue] = useState("");

  const {
    data: file,
    isLoading,
    isError,
  } = useGetV2GetSandboxFile(
    sessionId,
    { path },
    { query: { select: (res) => res.data as SandboxFileResponse } },
  );

  useEffect(() => {
    if (file) setValue(file.content);
  }, [file]);

  const { mutateAsync: writeFile, isPending: isSaving } =
    usePutV2WriteSandboxFile({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2GetSandboxChangesQueryKey(sessionId),
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2GetSandboxFileQueryKey(sessionId, { path }),
          });
          toast({ title: "Saved", variant: "success" });
        },
        onError: (error) => {
          toast({
            title: "Save failed",
            description: error instanceof Error ? error.message : undefined,
            variant: "destructive",
          });
        },
      },
    });

  async function save() {
    if (file?.truncated) return;
    await writeFile({ sessionId, data: { path, content: value } });
  }

  return {
    value,
    setValue,
    save,
    isLoading,
    isError,
    isSaving,
    truncated: file?.truncated ?? false,
  };
}
