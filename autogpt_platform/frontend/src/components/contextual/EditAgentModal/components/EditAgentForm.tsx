"use client";

import * as React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Form, FormField } from "@/components/ui/form";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { ThumbnailImages } from "../../PublishAgentModal/components/AgentInfoStep/components/ThumbnailImages";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { StepHeader } from "../../PublishAgentModal/components/StepHeader";
import { useEditAgentForm } from "./useEditAgentForm";

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
  const {
    form,
    categoryOptions,
    isSubmitting,
    handleFormSubmit,
    handleImagesChange,
  } = useEditAgentForm({ submission, onSuccess });

  return (
    <div className="mx-auto flex w-full flex-col rounded-3xl">
      <StepHeader title="Edit Agent" description="Update your agent details" />

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

          <FormField
            control={form.control}
            name="changes_summary"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Changes Summary"
                type="text"
                placeholder="Briefly describe what you changed"
                error={form.formState.errors.changes_summary?.message}
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
