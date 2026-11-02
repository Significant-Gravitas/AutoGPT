import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  getListExpertIdentitiesQueryKey,
  useUpdateExpertAvatar,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useExpertAvatarButton(expertId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const { mutateAsync: updateAvatar, isPending } = useUpdateExpertAvatar();

  async function saveAvatar(url: string) {
    try {
      await updateAvatar({ expertId, data: { avatar_url: url } });
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expertId),
        }),
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() }),
        queryClient.invalidateQueries({
          queryKey: getListExpertIdentitiesQueryKey(),
        }),
      ]);
      setIsOpen(false);
      toast({ title: "Avatar updated", variant: "success" });
    } catch {
      toast({
        title: "Could not update avatar",
        description: "Your current avatar is unchanged. Try again.",
        variant: "destructive",
      });
    }
  }

  return { isOpen, setIsOpen, saveAvatar, isPending };
}
