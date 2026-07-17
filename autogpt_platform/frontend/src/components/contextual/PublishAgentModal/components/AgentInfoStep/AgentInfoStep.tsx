"use client";

import { CronExpressionDialog } from "@/components/contextual/CronScheduler/cron-scheduler-dialog";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { cn } from "@/lib/utils";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  CalendarDotsIcon,
  CheckCircleIcon,
  ImagesIcon,
  InfoIcon,
  SparkleIcon,
  StorefrontIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";
import * as React from "react";
import { StepHeader } from "../StepHeader";
import { StepFooter } from "../StepFooter";
import { ThumbnailImages } from "./components/ThumbnailImages";
import { CharCountedTextarea } from "./components/CharCountedTextarea";
import { Props, useAgentInfoStep } from "./useAgentInfoStep";

const DESCRIPTION_MAX = 1000;
const INSTRUCTIONS_MAX = 2000;

export function AgentInfoStep({
  onBack,
  onSuccess,
  selectedAgentId,
  selectedAgentVersion,
  initialData,
  isMarketplaceUpdate,
}: Props) {
  const {
    form,
    agentId,
    images,
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
    isMarketplaceUpdate,
  });

  const [cronScheduleDialogOpen, setCronScheduleDialogOpen] =
    React.useState(false);
  const [openAccordion, setOpenAccordion] = React.useState("");
  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      setOpenAccordion((current) => current || "basics");
    }, 320);
    return () => window.clearTimeout(timer);
  }, []);

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

  const isSubmitDisabled =
    Object.keys(form.formState.errors).length > 0 || isSubmitting;

  const watched = form.watch();
  const errors = form.formState.errors;
  const basicsHasError = !!(
    errors.title ||
    errors.slug ||
    errors.subheader ||
    errors.category
  );
  const experienceHasError = !!(
    errors.description ||
    errors.instructions ||
    errors.youtubeLink ||
    errors.agentOutputDemo
  );
  const thumbnailsHasError = !!errors.root;
  const basicsComplete =
    !basicsHasError &&
    !!watched.title &&
    !!watched.slug &&
    !!watched.subheader &&
    !!watched.category;
  const thumbnailsComplete = !thumbnailsHasError && images.length > 0;
  const experienceComplete =
    !experienceHasError &&
    !!watched.description &&
    !!watched.youtubeLink &&
    !!watched.agentOutputDemo;

  const title = isMarketplaceUpdate
    ? "Describe the update"
    : "Build the store listing";
  const description = isMarketplaceUpdate
    ? "Explain what changed, then adjust any listing details that need to move with this version."
    : "Write a bit of details about your agent so reviewers and users know what to expect.";

  return (
    <div className="mx-auto flex w-full flex-col">
      <StepHeader title={title} description={description} currentStep="info" />

      <Form {...form}>
        <form onSubmit={handleSubmit} className="flex flex-col gap-5 pb-5">
          {isMarketplaceUpdate && (
            <section className="rounded-[18px] border border-amber-200 bg-amber-50 p-4">
              <div className="mb-4 flex items-start gap-3">
                <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-white text-amber-700 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
                  <InfoIcon size={18} weight="duotone" />
                </div>
                <div className="flex min-w-0 flex-col gap-1">
                  <Text
                    variant="body-medium"
                    as="h3"
                    className="text-amber-950"
                  >
                    Update note
                  </Text>
                  <Text variant="small" className="text-amber-800">
                    Reviewers use this to understand why the marketplace listing
                    needs a new version.
                  </Text>
                </div>
              </div>
              <FormField
                control={form.control}
                name="changesSummary"
                render={({ field }) => (
                  <Input
                    id={field.name}
                    labelVariant="body"
                    label="What changed?"
                    labelTooltip="Summary of what's new or improved in this version. Reviewers see this first."
                    type="textarea"
                    rows={1}
                    placeholder="Describe what's new or improved in this version..."
                    error={form.formState.errors.changesSummary?.message}
                    required
                    wrapperClassName="!mb-0"
                    {...field}
                  />
                )}
              />
            </section>
          )}

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
                  name="slug"
                  render={({ field }) => (
                    <Input
                      id={field.name}
                      labelVariant="body"
                      label="Slug"
                      labelTooltip="URL-friendly identifier used in the marketplace path. Lowercase letters, numbers, hyphens only."
                      type="text"
                      placeholder="URL-friendly name"
                      error={form.formState.errors.slug?.message}
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
                  agentId={agentId}
                  onImagesChange={handleImagesChange}
                  initialImages={initialImages}
                  initialSelectedImage={initialSelectedImage}
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

                <FormField
                  control={form.control}
                  name="instructions"
                  render={({ field }) => (
                    <CharCountedTextarea
                      max={INSTRUCTIONS_MAX}
                      value={field.value ?? ""}
                    >
                      <Input
                        id={field.name}
                        labelVariant="body"
                        label="Instructions"
                        labelTooltip="Steps users should follow to set up and run the agent."
                        type="textarea"
                        rows={2}
                        placeholder="Explain inputs, setup, and what to expect after a run"
                        error={form.formState.errors.instructions?.message}
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

                <FormField
                  control={form.control}
                  name="recommendedScheduleCron"
                  render={({ field }) => (
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-1">
                          <Text variant="body" as="span" className="text-black">
                            Recommended schedule
                          </Text>
                          <InformationTooltip
                            description="Suggested cron schedule users can run the agent on. Sets a run cadence for recurring use cases."
                            iconSize={20}
                          />
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => setCronScheduleDialogOpen(true)}
                        className="flex h-[2.875rem] w-full items-center gap-2 rounded-xl border border-zinc-200 bg-white px-4 py-2.5 text-left text-sm font-normal text-black shadow-none transition-colors hover:border-zinc-300 focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-400"
                      >
                        <CalendarDotsIcon
                          size={16}
                          weight="duotone"
                          className="shrink-0 text-zinc-500"
                        />
                        <span
                          className={cn(
                            "truncate",
                            !field.value && "text-zinc-400",
                          )}
                        >
                          {field.value
                            ? humanizeCronExpression(field.value)
                            : "Set schedule"}
                        </span>
                      </button>
                    </div>
                  )}
                />
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <StepFooter
            secondary={
              <Button
                type="button"
                onClick={onBack}
                variant="secondary"
                size="small"
                className="w-full sm:w-auto"
              >
                Back
              </Button>
            }
            primary={
              <Button
                type="submit"
                size="small"
                disabled={isSubmitDisabled}
                loading={isSubmitting}
                className="w-full sm:w-auto"
              >
                {isSubmitting ? "Submitting" : "Submit for review"}
              </Button>
            }
          />
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
