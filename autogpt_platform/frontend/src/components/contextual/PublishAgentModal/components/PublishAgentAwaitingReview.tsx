"use client";

import * as React from "react";

import Image from "next/image";
import { StepHeader } from "./StepHeader";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  agentName: string;
  subheader: string;
  description: string;
  thumbnailSrc?: string;
  onClose: () => void;
  onDone: () => void;
  onViewProgress: () => void;
}

export function PublishAgentAwaitingReview({
  agentName,
  subheader,
  description,
  thumbnailSrc,
  onDone,
  onViewProgress,
}: Props) {
  return (
    <div aria-labelledby="modal-title">
      <StepHeader
        title="Agent is awaiting review"
        description="In the meantime you can check your progress on your Creator Dashboard page"
      />

      <div className="flex flex-1 flex-col items-center gap-8 px-6 pt-6 sm:gap-6">
        <div className="mt-4 flex w-full flex-col items-center gap-6 sm:mt-0 sm:gap-4">
          <div className="flex flex-col items-center gap-3 sm:gap-2">
            <Text variant="large-medium">{agentName}</Text>
            <Text variant="body" className="!text-neutral-500">
              {subheader}
            </Text>
          </div>

          <div
            className="h-[280px] w-full rounded-xl bg-neutral-200 dark:bg-neutral-700"
            role="img"
            aria-label={
              thumbnailSrc ? "Agent thumbnail" : "Thumbnail placeholder"
            }
          >
            {thumbnailSrc && (
              <Image
                src={thumbnailSrc}
                alt="Agent thumbnail"
                width={400}
                height={280}
                className="h-full w-full rounded-xl object-cover"
              />
            )}
          </div>

          {description ? (
            <Text
              variant="body-medium"
              className="pb-16 pt-4 !text-neutral-500"
            >
              {description}
            </Text>
          ) : null}
        </div>
      </div>
      <div className="flex justify-between gap-4">
        <Button variant="secondary" onClick={onDone} className="w-full">
          Done
        </Button>
        <Button onClick={onViewProgress} className="w-full">
          View progress
        </Button>
      </div>
    </div>
  );
}
