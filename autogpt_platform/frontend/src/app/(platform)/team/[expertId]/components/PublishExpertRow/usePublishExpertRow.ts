import {
  getGetPublishedTemplateQueryKey,
  getListExpertTemplatesQueryKey,
  publishExpert,
  useGetPublishedTemplate,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackExpertPublished } from "@/services/experts/experts-analytics";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { buildPreviewFromExpert, getPublishErrorMessage } from "./helpers";

interface Args {
  expert: Expert;
  enabled: boolean;
}

export function usePublishExpertRow({ expert, enabled }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = useState(false);
  const [isPublishing, setIsPublishing] = useState(false);

  const { data: published } = useGetPublishedTemplate(expert.id, {
    query: {
      enabled,
      select: (response) => (response.status === 200 ? response.data : null),
    },
  });

  async function confirmPublish() {
    setIsPublishing(true);
    try {
      const response = await publishExpert(expert.id);
      if (response.status !== 201) {
        throw new Error(`Couldn't publish ${expert.name}`);
      }

      toast({ title: `Published ${expert.name}` });
      trackExpertPublished({
        expert_id: expert.id,
        workflow_count: expert.workflows.length,
        skill_count: expert.skills.length,
      });
      setIsOpen(false);
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getGetPublishedTemplateQueryKey(expert.id),
        }),
        queryClient.invalidateQueries({
          queryKey: getListExpertTemplatesQueryKey(),
        }),
      ]);
    } catch (error) {
      toast({
        title: `Couldn't publish ${expert.name}`,
        description: getPublishErrorMessage(error, expert.name),
        variant: "destructive",
      });
    } finally {
      setIsPublishing(false);
    }
  }

  return {
    isLive: Boolean(published),
    isOpen,
    isPublishing,
    preview: isOpen ? buildPreviewFromExpert(expert) : null,
    openDialog: () => setIsOpen(true),
    closeDialog: () => setIsOpen(false),
    confirmPublish,
  };
}
