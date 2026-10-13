import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { usePostV2SaveUiDraftToFile } from "@/app/api/__generated__/endpoints/chat/chat";
import { getListWorkspaceFilesQueryKey } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { invalidateWorkspaceFileQueries } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { validateFileName } from "@/app/(platform)/artifacts/components/RenameFileDialog/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { defaultViewFileName } from "@/lib/openui/file";
import type { SavedFileRef } from "../../helpers";

interface Args {
  ownerId: string | undefined;
  sessionId: string;
  viewId: string;
  title: string;
  beforeSave: () => Promise<void>;
  onSaved: (file: SavedFileRef) => void;
}

export function useSaveViewDialog(args: Args) {
  const queryClient = useQueryClient();
  const [name, setName] = useState(defaultViewFileName(args.title));
  const [error, setError] = useState<string | null>(null);
  const { mutateAsync, isPending } = usePostV2SaveUiDraftToFile();
  const trimmed = name.trim();
  const validationError = validateFileName(trimmed);

  async function handleSubmit() {
    if (validationError || isPending) return;
    setError(null);
    try {
      await args.beforeSave();
      const result = await mutateAsync({
        sessionId: args.sessionId,
        viewId: args.viewId,
        data: { filename: trimmed },
      });
      if (result.status !== 200) throw new Error("save");
      void queryClient.invalidateQueries({
        queryKey: ["ui-draft", args.ownerId, args.sessionId, args.viewId],
      });
      void queryClient.invalidateQueries({
        queryKey: getListWorkspaceFilesQueryKey(),
      });
      invalidateWorkspaceFileQueries(queryClient);
      args.onSaved({ id: result.data.file_id, name: result.data.file_name });
    } catch (cause) {
      setError(
        cause instanceof ApiError && cause.status === 409
          ? "A file with this name already exists. Choose another name."
          : "The view couldn't be saved. Try again.",
      );
    }
  }

  return {
    name,
    setName,
    error: trimmed.length > 0 ? (validationError ?? error) : error,
    canSubmit: !validationError && !isPending,
    isPending,
    handleSubmit,
  };
}
