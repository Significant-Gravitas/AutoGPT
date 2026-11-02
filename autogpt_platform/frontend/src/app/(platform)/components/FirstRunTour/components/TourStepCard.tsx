"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

interface Props {
  icon: IconSvgElement;
  title: string;
  body: string;
  titleID: string;
  bodyID: string;
  stepNumber: number;
  stepCount: number;
  isFirstStep: boolean;
  isLastStep: boolean;
  onBack: () => void;
  onNext: () => void;
  onSkip: () => void;
  /** A quiet second way forward, under the copy (last step only). */
  altAction?: { label: string; onClick: () => void };
}

export function TourStepCard({
  icon,
  title,
  body,
  titleID,
  bodyID,
  stepNumber,
  stepCount,
  isFirstStep,
  isLastStep,
  onBack,
  onNext,
  onSkip,
  altAction,
}: Props) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2.5">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border-base bg-surface-muted text-foreground-emphasis">
          <Icon icon={icon} size={16} aria-hidden />
        </span>
        <StepProgress stepNumber={stepNumber} stepCount={stepCount} />
      </div>
      <div className="flex flex-col gap-1">
        <Text
          as="h2"
          id={titleID}
          variant="large-semibold"
          tone="primary"
          className="text-balance [overflow-wrap:anywhere]"
        >
          {title}
        </Text>
        <Text
          id={bodyID}
          variant="body"
          tone="secondary"
          className="text-pretty [overflow-wrap:anywhere]"
        >
          {body}
        </Text>
        {altAction && (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={altAction.onClick}
            className="-ml-2.5 w-fit text-foreground-emphasis underline decoration-border-strong underline-offset-4 hover:bg-transparent hover:decoration-foreground-emphasis"
          >
            {altAction.label}
          </Button>
        )}
      </div>
      <div className="mt-1 flex items-center justify-between gap-2">
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={onSkip}
          className="-ml-2.5 text-foreground-secondary"
        >
          {isLastStep ? "Close" : "Skip tour"}
        </Button>
        <div className="flex items-center gap-2">
          {!isFirstStep && (
            <Button
              type="button"
              variant="secondary"
              size="xs"
              onClick={onBack}
            >
              Back
            </Button>
          )}
          <Button
            data-tour-primary
            type="button"
            variant="primary"
            size="xs"
            onClick={onNext}
            rightIcon={
              isLastStep ? (
                <Icon icon={ArrowRight01Icon} size={14} aria-hidden />
              ) : undefined
            }
          >
            {isLastStep ? "Start here" : "Next"}
          </Button>
        </div>
      </div>
    </div>
  );
}

interface StepProgressProps {
  stepNumber: number;
  stepCount: number;
}

function StepProgress({ stepNumber, stepCount }: StepProgressProps) {
  return (
    <div className="flex items-center gap-2.5">
      <div aria-hidden className="flex items-center gap-1">
        {Array.from({ length: stepCount }, (_, index) => (
          <span
            key={index}
            className={cn(
              "h-1 rounded-full transition-[width,background-color] duration-300 ease-out motion-reduce:transition-none",
              index + 1 === stepNumber ? "w-4 bg-foreground-emphasis" : "w-1.5",
              index + 1 < stepNumber && "bg-foreground-tertiary",
              index + 1 > stepNumber && "bg-surface-active",
            )}
          />
        ))}
      </div>
      <Text
        variant="small"
        tone="secondary"
        className="tabular-nums"
        dir="auto"
      >
        {stepNumber} of {stepCount}
      </Text>
    </div>
  );
}
