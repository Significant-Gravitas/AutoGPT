import {
  getListCopilotSkillsQueryKey,
  useDeleteCopilotSkill,
} from "@/app/api/__generated__/endpoints/skills/skills";
import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { describeSkill } from "./helpers";

interface Args {
  skill: CopilotSkillInfo;
}

export function useSkillListItem({ skill }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

  const { mutateAsync: deleteSkill, isPending: isDeleting } =
    useDeleteCopilotSkill();

  const { descriptionPreview, triggers } = describeSkill(skill);

  function openDelete() {
    setIsDeleteOpen(true);
  }

  function closeDelete(open: boolean) {
    setIsDeleteOpen(open);
  }

  async function handleDelete() {
    try {
      await deleteSkill({ name: skill.name });
      toast({ title: "Skill deleted" });
      setIsDeleteOpen(false);
      queryClient.invalidateQueries({
        queryKey: getListCopilotSkillsQueryKey(),
      });
    } catch (error) {
      toast({
        title: "Failed to delete skill",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return {
    descriptionPreview,
    triggers,
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
  };
}
