import { useEffect, useCallback, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  PublishAgentFormData,
  PublishAgentInfoInitialData,
  publishAgentSchema,
} from "./helpers";

export interface Props {
  onBack: () => void;
  onSubmit: (
    name: string,
    subHeading: string,
    slug: string,
    description: string,
    imageUrls: string[],
    videoUrl: string,
    categories: string[],
  ) => void;
  initialData?: PublishAgentInfoInitialData;
}

export function usePublishAgentInfo({
  onBack: _onBack,
  onSubmit,
  initialData,
}: Props) {
  const [agentId, setAgentId] = useState<string | null>(null);
  const [images, setImages] = useState<string[]>([]);

  const form = useForm<PublishAgentFormData>({
    resolver: zodResolver(publishAgentSchema),
    defaultValues: {
      title: "",
      subheader: "",
      slug: "",
      youtubeLink: "",
      category: "",
      description: "",
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
        slug: initialData.slug,
        youtubeLink: initialData.youtubeLink,
        category: initialData.category,
        description: initialData.description,
      });
    }
  }, [initialData, form]);

  const handleImagesChange = useCallback((newImages: string[]) => {
    setImages(newImages);
  }, []);

  function handleFormSubmit(data: PublishAgentFormData) {
    // Validate that at least one image is present
    if (images.length === 0) {
      form.setError("root", {
        type: "manual",
        message: "At least one image is required",
      });
      return;
    }

    const categories = data.category ? [data.category] : [];
    onSubmit(
      data.title,
      data.subheader,
      data.slug,
      data.description,
      images,
      data.youtubeLink || "",
      categories,
    );
  }

  return {
    form,
    agentId,
    images,
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
