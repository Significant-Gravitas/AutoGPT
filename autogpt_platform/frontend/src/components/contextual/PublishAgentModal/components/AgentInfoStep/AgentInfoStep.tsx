"use client";

import { CronExpressionDialog } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/components/cron-scheduler-dialog";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { CalendarClockIcon } from "lucide-react";
import * as React from "react";
import { StepHeader } from "../StepHeader";
import { ThumbnailImages } from "./components/ThumbnailImages";
import { Props, useAgentInfoStep } from "./useAgentInfoStep";

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

  const [cronScheduleDialogOpen, setCronScheduleDialogOpen] =
    React.useState(false);

  const handleScheduleChange = (cronExpression: string) => {
    form.setValue("recommendedScheduleCron", cronExpression);
  };

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

          <FormField
            control={form.control}
            name="agentOutputDemo"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Agent Output Demo"
                type="url"
                placeholder="Add a short video showing the agent's results in action."
                error={form.formState.errors.agentOutputDemo?.message}
                {...field}
              />
            )}
          />

          <FormField
            control={form.control}
            name="instructions"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Instructions"
                type="textarea"
                placeholder="Explain to users how to run this agent and what to expect"
                error={form.formState.errors.instructions?.message}
                {...field}
              />
            )}
          />

          <FormField
            control={form.control}
            name="recommendedScheduleCron"
            render={({ field }) => (
              <div className="mb-8 flex flex-col space-y-2">
                <Text variant="large-medium">Recommended Schedule</Text>
                <p className="text-xs text-gray-600">
                  Suggest when users should run this agent for best results
                </p>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setCronScheduleDialogOpen(true)}
                  className="w-full justify-start text-sm"
                >
                  <CalendarClockIcon className="mr-2 h-4 w-4" />
                  {field.value
                    ? humanizeCronExpression(field.value)
                    : "Set schedule"}
                </Button>
              </div>
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

      <CronExpressionDialog
        open={cronScheduleDialogOpen}
        setOpen={setCronScheduleDialogOpen}
        onSubmit={handleScheduleChange}
        defaultCronExpression={form.getValues("recommendedScheduleCron") || ""}
        title="Recommended Schedule"
      />
    </div>
  );
}
