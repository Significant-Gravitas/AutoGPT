"use client";

import * as React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { StepHeader } from "../StepHeader";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Form, FormField } from "@/components/ui/form";
import { Props, useAgentInfoStep } from "./useAgentInfoStep";
import { ThumbnailImages } from "./components/ThumbnailImages";

export function AgentInfoStep({
  onBack,
  onSuccess,
  selectedAgentId,
  selectedAgentVersion,
  initialData,
}: Props) {
  const {
    form,
    agentId,
    initialImages,
    initialSelectedImage,
    handleImagesChange,
    handleSubmit,
    isSubmitting,
  } = useAgentInfoStep({
    onBack,
    onSuccess,
    selectedAgentId,
    selectedAgentVersion,
    initialData,
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

  return (
    <div className="mx-auto flex w-full flex-col rounded-3xl">
      <StepHeader
        title="Publish Agent"
        description="Write a bit of details about your agent"
      />

      <Form {...form}>
        <form onSubmit={handleSubmit} className="flex-grow overflow-y-auto p-6">
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

          <FormField
            control={form.control}
            name="slug"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Slug"
                type="text"
                placeholder="URL-friendly name for your agent"
                error={form.formState.errors.slug?.message}
                {...field}
              />
            )}
          />

          <ThumbnailImages
            agentId={agentId}
            onImagesChange={handleImagesChange}
            initialImages={initialImages}
            initialSelectedImage={initialSelectedImage}
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
            render={({ field }) => (
              <Select
                id={field.name}
                label="Category"
                placeholder="Select a category for your agent"
                value={field.value}
                onValueChange={field.onChange}
                error={form.formState.errors.category?.message}
                options={categoryOptions}
              />
            )}
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
              onClick={onBack}
              variant="secondary"
              className="w-full"
            >
              Back
            </Button>
            <Button
              type="submit"
              className="w-full"
              disabled={
                Object.keys(form.formState.errors).length > 0 || isSubmitting
              }
              loading={isSubmitting}
            >
              Submit for review
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
