import {
  getListCopilotSkillsQueryKey,
  useUploadCopilotSkill,
} from "@/app/api/__generated__/endpoints/skills/skills";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";

export function useUploadSkillButton() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { mutateAsync: uploadSkill, isPending: isUploading } =
    useUploadCopilotSkill();

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    // Reset the input so re-picking the same file fires onChange again.
    event.target.value = "";
    if (!file) return;

    try {
      const content = await file.text();
      const result = await uploadSkill({ data: { content } });
      const name =
        result.status === 201 ? result.data.name : (file.name ?? "skill");
      toast({ title: `Skill "${name}" uploaded` });
      queryClient.invalidateQueries({
        queryKey: getListCopilotSkillsQueryKey(),
      });
    } catch (error) {
      toast({
        title: "Failed to upload skill",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return {
    fileInputRef,
    isUploading,
    openFilePicker,
    handleFileChange,
  };
}
