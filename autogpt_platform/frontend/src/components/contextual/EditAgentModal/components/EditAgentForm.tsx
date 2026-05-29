"use client";

import * as React from "react";
import {
  CheckCircleIcon,
  ImagesIcon,
  InfoIcon,
  SparkleIcon,
  StorefrontIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";

import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";

import { CharCountedTextarea } from "../../PublishAgentModal/components/AgentInfoStep/components/CharCountedTextarea";
import { ThumbnailImages } from "../../PublishAgentModal/components/AgentInfoStep/components/ThumbnailImages";

import { useEditAgentForm } from "./useEditAgentForm";

const DESCRIPTION_MAX = 1000;
const CHANGES_SUMMARY_MAX = 500;

interface EditAgentFormProps {
  submission: StoreSubmissionEditRequest & {
    store_listing_version_id: string | undefined;
    graph_id: string;
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
    images,
    categoryOptions,
    isSubmitting,
    handleFormSubmit,
    handleImagesChange,
  } = useEditAgentForm({ submission, onSuccess, onClose });

  const [openAccordion, setOpenAccordion] = React.useState("basics");

  const watched = form.watch();
  const errors = form.formState.errors;

  const basicsHasError = !!(
    errors.title ||
    errors.subheader ||
    errors.category
  );
  const experienceHasError = !!(
    errors.description ||
    errors.youtubeLink ||
    errors.agentOutputDemo
  );
  const thumbnailsHasError = !!errors.root;

  const basicsComplete =
    !basicsHasError &&
    !!watched.title &&
    !!watched.subheader &&
    !!watched.category;
  const experienceComplete = !experienceHasError && !!watched.description;
  const thumbnailsComplete = !thumbnailsHasError && images.length > 0;

  const isSubmitDisabled =
    Object.keys(form.formState.errors).length > 0 || isSubmitting;

  return (
    <div className="mx-auto flex w-full flex-col">
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(handleFormSubmit)}
          className="flex flex-col gap-5 pb-5"
        >
          <section className="rounded-[18px] border border-amber-200 bg-amber-50 p-4">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-white text-amber-700 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
                <InfoIcon size={18} weight="duotone" />
              </div>
              <div className="flex min-w-0 flex-col gap-1">
                <Text variant="body-medium" as="h3" className="text-amber-950">
                  Update note
                </Text>
                <Text variant="small" className="text-amber-800">
                  Reviewers use this to understand why this submission was
                  edited.
                </Text>
              </div>
            </div>
            <FormField
              control={form.control}
              name="changes_summary"
              render={({ field }) => (
                <CharCountedTextarea
                  max={CHANGES_SUMMARY_MAX}
                  value={field.value ?? ""}
                >
                  <Input
                    id={field.name}
                    labelVariant="body"
                    label="What changed?"
                    labelTooltip="Summary of what's new or improved in this version. Reviewers see this first."
                    type="textarea"
                    rows={1}
                    placeholder="Briefly describe what you changed"
                    error={form.formState.errors.changes_summary?.message}
                    required
                    wrapperClassName="!mb-0"
                    {...field}
                  />
                </CharCountedTextarea>
              )}
            />
          </section>

          <Accordion
            type="single"
            collapsible
            value={openAccordion}
            onValueChange={setOpenAccordion}
            className="overflow-hidden rounded-[14px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)] [&>*+*]:border-t [&>*+*]:border-zinc-200"
          >
            <AccordionItem value="basics" className="border-0 px-4">
              <AccordionTrigger className="hover:no-underline">
                <span className="flex items-center gap-2 text-sm font-medium text-textBlack">
                  <StorefrontIcon
                    size={18}
                    weight="duotone"
                    className="text-zinc-500"
                  />
                  Listing basics
                  {basicsHasError ? (
                    <WarningCircleIcon
                      size={16}
                      weight="fill"
                      className="text-rose-500"
                    />
                  ) : basicsComplete ? (
                    <CheckCircleIcon
                      size={16}
                      weight="fill"
                      className="text-purple-500"
                    />
                  ) : null}
                </span>
              </AccordionTrigger>
              <AccordionContent className="px-1 pb-4 pt-0">
                <div className="grid gap-x-4 sm:grid-cols-2">
                  <FormField
                    control={form.control}
                    name="title"
                    render={({ field }) => (
                      <Input
                        id={field.name}
                        labelVariant="body"
                        label="Title"
                        labelTooltip="Public name shown on the marketplace listing."
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
                        labelVariant="body"
                        label="Subheader"
                        labelTooltip="One-sentence tagline displayed under the title."
                        type="text"
                        placeholder="A concise tagline for your agent"
                        error={form.formState.errors.subheader?.message}
                        {...field}
                      />
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="category"
                  render={({ field }) => (
                    <Select
                      id={field.name}
                      labelVariant="body"
                      label="Category"
                      labelTooltip="Primary category that helps users discover the agent."
                      placeholder="Select a category"
                      value={field.value}
                      onValueChange={field.onChange}
                      error={form.formState.errors.category?.message}
                      options={categoryOptions}
                    />
                  )}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="thumbnails" className="border-0 px-4">
              <AccordionTrigger className="hover:no-underline">
                <span className="flex items-center gap-2 text-sm font-medium text-textBlack">
                  <ImagesIcon
                    size={18}
                    weight="duotone"
                    className="text-zinc-500"
                  />
                  Thumbnails
                  {thumbnailsHasError ? (
                    <WarningCircleIcon
                      size={16}
                      weight="fill"
                      className="text-rose-500"
                    />
                  ) : thumbnailsComplete ? (
                    <CheckCircleIcon
                      size={16}
                      weight="fill"
                      className="text-purple-500"
                    />
                  ) : null}
                </span>
              </AccordionTrigger>
              <AccordionContent className="px-1 pb-4 pt-0">
                <ThumbnailImages
                  agentId={submission.graph_id}
                  onImagesChange={handleImagesChange}
                  initialImages={Array.from(
                    new Set(submission.image_urls || []),
                  )}
                  initialSelectedImage={submission.image_urls?.[0] || null}
                  errorMessage={form.formState.errors.root?.message}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="experience" className="border-0 px-4">
              <AccordionTrigger className="hover:no-underline">
                <span className="flex items-center gap-2 text-sm font-medium text-textBlack">
                  <SparkleIcon
                    size={18}
                    weight="duotone"
                    className="text-zinc-500"
                  />
                  Experience details
                  {experienceHasError ? (
                    <WarningCircleIcon
                      size={16}
                      weight="fill"
                      className="text-rose-500"
                    />
                  ) : experienceComplete ? (
                    <CheckCircleIcon
                      size={16}
                      weight="fill"
                      className="text-purple-500"
                    />
                  ) : null}
                </span>
              </AccordionTrigger>
              <AccordionContent className="px-1 pb-4 pt-0">
                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <CharCountedTextarea
                      max={DESCRIPTION_MAX}
                      value={field.value ?? ""}
                    >
                      <Input
                        id={field.name}
                        labelVariant="body"
                        label="Description"
                        labelTooltip="What the agent does and the outcome users get."
                        type="textarea"
                        rows={2}
                        placeholder="Describe the outcome this agent creates"
                        error={form.formState.errors.description?.message}
                        {...field}
                      />
                    </CharCountedTextarea>
                  )}
                />

                <div className="grid gap-x-4 sm:grid-cols-2">
                  <FormField
                    control={form.control}
                    name="youtubeLink"
                    render={({ field }) => (
                      <Input
                        id={field.name}
                        labelVariant="body"
                        label="YouTube video link"
                        labelTooltip="Demo or walkthrough video hosted on YouTube."
                        type="url"
                        placeholder="https://youtube.com/watch?v=..."
                        error={form.formState.errors.youtubeLink?.message}
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
                        labelVariant="body"
                        label="Output demo"
                        labelTooltip="Link showing example output the agent produces."
                        type="url"
                        placeholder="https://youtube.com/watch?v=..."
                        error={form.formState.errors.agentOutputDemo?.message}
                        {...field}
                      />
                    )}
                  />
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="flex flex-col-reverse gap-3 border-t border-zinc-200 pt-4 sm:flex-row sm:justify-end">
            <Button
              type="button"
              onClick={onClose}
              variant="secondary"
              size="small"
              className="w-full sm:w-auto"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              size="small"
              disabled={isSubmitDisabled}
              loading={isSubmitting}
              className="w-full sm:w-auto"
            >
              {isSubmitting ? "Saving" : "Update submission"}
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
