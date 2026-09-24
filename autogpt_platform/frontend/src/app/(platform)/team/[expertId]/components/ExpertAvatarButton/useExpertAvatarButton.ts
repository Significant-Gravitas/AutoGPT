import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  useUpdateExpertAvatar,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  isFileTooLarge,
  uploadSubmissionMediaDirect,
} from "@/lib/direct-upload";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useExpertAvatarButton(expertId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isUploading, setIsUploading] = useState(false);

  const { mutateAsync: updateAvatar } = useUpdateExpertAvatar({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expertId),
        });
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
      },
    },
  });

  async function uploadAvatar(file: File) {
    if (isFileTooLarge({ file, maxSizeMB: 5, toast })) return;

    setIsUploading(true);
    try {
      const url = (
        await uploadSubmissionMediaDirect(file, "expert-avatar")
      ).trim();
      if (!url) throw new Error("Upload returned no URL");
      await updateAvatar({ expertId, data: { avatar_url: url } });
      toast({ title: "Appearance updated", variant: "success" });
    } catch (error) {
      toast({
        title: "Couldn't update appearance",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  }

  return { uploadAvatar, isUploading };
}
