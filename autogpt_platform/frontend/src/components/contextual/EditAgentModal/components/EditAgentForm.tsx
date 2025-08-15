"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  getGetV2ListMySubmissionsQueryKey,
  usePutV2EditStoreSubmission,
} from "@/app/api/__generated__/endpoints/store/store";
import * as Sentry from "@sentry/nextjs";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Form, FormField } from "@/components/ui/form";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { ThumbnailImages } from "../../PublishAgentModal/components/AgentInfoStep/components/ThumbnailImages";
import { z } from "zod";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";

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
});

type EditAgentFormData = z.infer<typeof editAgentSchema>;

interface EditAgentFormProps {
  submission: StoreSubmissionEditRequest & {
    store_listing_version_id: string | undefined;
    agent_id: string;
  };
  onClose: () => void;
  onSuccess: (submission: StoreSubmission) => void;
}

export function EditAgentForm({
  submission,
  onClose,
  onSuccess,
}: EditAgentFormProps) {
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
          changes_summary: "Updated submission",
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

  return (
    <div className="mx-auto flex w-full flex-col rounded-3xl">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Edit Agent</h2>
        <p className="text-gray-600">Update your agent details</p>
      </div>

      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(handleFormSubmit)}
          className="flex-grow overflow-y-auto p-6"
        >
          <FormField
            control={form.control}
            name="title"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Title"
                type="text"
                placeholder="Agent name"
                error={form.formState.errors.title?.message}
                {...field}
              />
            )}
          />

          <FormField
            control={form.control}
            name="subheader"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Subheader"
                type="text"
                placeholder="A tagline for your agent"
                error={form.formState.errors.subheader?.message}
                {...field}
              />
            )}
          />

          <ThumbnailImages
            agentId={submission.agent_id}
            onImagesChange={handleImagesChange}
            initialImages={submission.image_urls || []}
            initialSelectedImage={submission.image_urls?.[0] || null}
            errorMessage={form.formState.errors.root?.message}
          />

          <FormField
            control={form.control}
            name="youtubeLink"
            render={({ field }) => (
              <Input
                id={field.name}
                label="YouTube video link"
                type="url"
                placeholder="Paste a video link here"
                error={form.formState.errors.youtubeLink?.message}
                {...field}
              />
            )}
          />

          <FormField
            control={form.control}
            name="category"
            render={({ field }) => {
              console.log("Edit Category field value:", field.value);
              return (
                <Select
                  id={field.name}
                  label="Category"
                  placeholder="Select a category for your agent"
                  value={field.value}
                  onValueChange={field.onChange}
                  error={form.formState.errors.category?.message}
                  options={categoryOptions}
                />
              );
            }}
          />

          <FormField
            control={form.control}
            name="description"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Description"
                type="textarea"
                placeholder="Describe your agent and what it does"
                error={form.formState.errors.description?.message}
                {...field}
              />
            )}
          />

          <div className="flex justify-between gap-4 pt-6">
            <Button
              type="button"
              onClick={onClose}
              variant="secondary"
              className="w-full"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              className="w-full"
              disabled={
                Object.keys(form.formState.errors).length > 0 || isSubmitting
              }
              loading={isSubmitting}
            >
              Update submission
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
