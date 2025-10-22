import { useEffect, useCallback, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { getGetV2ListMySubmissionsQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import * as Sentry from "@sentry/nextjs";
import {
  PublishAgentFormData,
  PublishAgentInfoInitialData,
  publishAgentSchema,
} from "./helpers";

export interface Props {
  onBack: () => void;
  onSuccess: (submissionData: any) => void;
  selectedAgentId: string | null;
  selectedAgentVersion: number | null;
  initialData?: PublishAgentInfoInitialData;
}

export function useAgentInfoStep({
  onBack: _onBack,
  onSuccess,
  selectedAgentId,
  selectedAgentVersion,
  initialData,
}: Props) {
  const [agentId, setAgentId] = useState<string | null>(null);
  const [images, setImages] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const queryClient = useQueryClient();
  const { toast } = useToast();
  const api = useBackendAPI();

  const form = useForm<PublishAgentFormData>({
    resolver: zodResolver(publishAgentSchema),
    defaultValues: {
      title: "",
      subheader: "",
      slug: "",
      youtubeLink: "",
      category: "",
      description: "",
      recommendedScheduleCron: "",
      instructions: "",
    },
  });

  useEffect(() => {
    if (initialData) {
      setAgentId(initialData.agent_id);
      const initialImages = [
        ...(initialData?.thumbnailSrc ? [initialData.thumbnailSrc] : []),
        ...(initialData.additionalImages || []),
      ];
      setImages(initialImages);

      // Update form with initial data
      form.reset({
        title: initialData.title,
        subheader: initialData.subheader,
        slug: initialData.slug.toLocaleLowerCase().trim(),
        youtubeLink: initialData.youtubeLink,
        category: initialData.category,
        description: initialData.description,
        recommendedScheduleCron: initialData.recommendedScheduleCron || "",
        instructions: initialData.instructions || "",
      });
    }
  }, [initialData, form]);

  const handleImagesChange = useCallback((newImages: string[]) => {
    setImages(newImages);
  }, []);

  async function handleFormSubmit(data: PublishAgentFormData) {
    // Validate that at least one image is present
    if (images.length === 0) {
      form.setError("root", {
        type: "manual",
        message: "At least one image is required",
      });
      return;
    }

    const categories = data.category ? [data.category] : [];
    const filteredCategories = categories.filter(Boolean);

    setIsSubmitting(true);

    try {
      const response = await api.createStoreSubmission({
        name: data.title,
        sub_heading: data.subheader,
        description: data.description,
        instructions: data.instructions || null,
        image_urls: images,
        video_url: data.youtubeLink || "",
        agent_id: selectedAgentId || "",
        agent_version: selectedAgentVersion || 0,
        slug: data.slug.replace(/\s+/g, "-"),
        categories: filteredCategories,
        recommended_schedule_cron: data.recommendedScheduleCron || null,
      });

      await queryClient.invalidateQueries({
        queryKey: getGetV2ListMySubmissionsQueryKey(),
      });

      onSuccess(response);
    } catch (error) {
      Sentry.captureException(error);
      toast({
        title: "Submit Agent Error",
        description:
          (error instanceof Error ? error.message : undefined) ||
          "An error occurred while submitting the agent. Please try again.",
        duration: 3000,
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return {
    form,
    agentId,
    images,
    isSubmitting,
    initialImages: initialData
      ? [
          ...(initialData?.thumbnailSrc ? [initialData.thumbnailSrc] : []),
          ...(initialData.additionalImages || []),
        ]
      : [],
    initialSelectedImage: initialData?.thumbnailSrc || null,
    handleImagesChange,
    handleSubmit: form.handleSubmit(handleFormSubmit),
  };
}
