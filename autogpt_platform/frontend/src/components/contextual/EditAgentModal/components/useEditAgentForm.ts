import * as Sentry from "@sentry/nextjs";
import {
  getGetV2ListMySubmissionsQueryKey,
  usePutV2EditStoreSubmission,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import React from "react";
import { useForm } from "react-hook-form";
import z from "zod";

interface useEditAgentFormProps {
  submission: StoreSubmissionEditRequest & {
    store_listing_version_id: string | undefined;
    agent_id: string;
  };
  onSuccess: (submission: StoreSubmission) => void;
}

export const useEditAgentForm = ({
  submission,
  onSuccess,
}: useEditAgentFormProps) => {
  const editAgentSchema = z.object({
    title: z
      .string()
      .min(1, "Title is required")
      .max(100, "Title must be less than 100 characters"),
    subheader: z
      .string()
      .min(1, "Subheader is required")
      .max(200, "Subheader must be less than 200 characters"),
    youtubeLink: z
      .string()
      .optional()
      .refine((val) => {
        if (!val) return true;
        try {
          const url = new URL(val);
          const allowedHosts = [
            "youtube.com",
            "www.youtube.com",
            "youtu.be",
            "www.youtu.be",
          ];
          return allowedHosts.includes(url.hostname);
        } catch {
          return false;
        }
      }, "Please enter a valid YouTube URL"),
    category: z.string().min(1, "Category is required"),
    description: z
      .string()
      .min(1, "Description is required")
      .max(1000, "Description must be less than 1000 characters"),
    changes_summary: z
      .string()
      .min(1, "Changes summary is required")
      .max(200, "Changes summary must be less than 200 characters"),
  });

  type EditAgentFormData = z.infer<typeof editAgentSchema>;

  const [images, setImages] = React.useState<string[]>(
    submission.image_urls || [],
  );
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const { mutateAsync: editSubmission } = usePutV2EditStoreSubmission({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListMySubmissionsQueryKey(),
        });
      },
    },
  });

  const queryClient = useQueryClient();
  const { toast } = useToast();

  const form = useForm<EditAgentFormData>({
    resolver: zodResolver(editAgentSchema),
    defaultValues: {
      title: submission.name,
      subheader: submission.sub_heading,
      youtubeLink: submission.video_url || "",
      category: submission.categories?.[0] || "",
      description: submission.description,
      changes_summary: submission.changes_summary || "",
    },
  });

  const categoryOptions = [
    { value: "productivity", label: "Productivity" },
    { value: "writing", label: "Writing & Content" },
    { value: "development", label: "Development" },
    { value: "data", label: "Data & Analytics" },
    { value: "marketing", label: "Marketing & SEO" },
    { value: "research", label: "Research & Learning" },
    { value: "creative", label: "Creative & Design" },
    { value: "business", label: "Business & Finance" },
    { value: "personal", label: "Personal Assistant" },
    { value: "other", label: "Other" },
  ];

  const handleImagesChange = React.useCallback((newImages: string[]) => {
    setImages(newImages);
  }, []);

  async function handleFormSubmit(data: EditAgentFormData) {
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
      const response = await editSubmission({
        storeListingVersionId: submission.store_listing_version_id!,
        data: {
          name: data.title,
          sub_heading: data.subheader,
          description: data.description,
          image_urls: images,
          video_url: data.youtubeLink || "",
          categories: filteredCategories,
          changes_summary: data.changes_summary,
        },
      });

      // Extract the StoreSubmission from the response
      if (response.status === 200 && response.data) {
        onSuccess(response.data);
      } else {
        throw new Error("Failed to update submission");
      }
    } catch (error) {
      Sentry.captureException(error);
      toast({
        title: "Edit Agent Error",
        description:
          "An error occurred while editing the agent. Please try again.",
        duration: 3000,
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return {
    form,
    isSubmitting,
    handleFormSubmit,
    handleImagesChange,
    categoryOptions,
  };
};
