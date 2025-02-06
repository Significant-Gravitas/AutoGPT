"use client";

import * as React from "react";
import { IconClose } from "../ui/icons";
import Image from "next/image";
import { Button } from "../agptui/Button";

interface PublishAgentAwaitingReviewProps {
  agentName: string;
  subheader: string;
  description: string;
  thumbnailSrc?: string;
  onClose: () => void;
  onDone: () => void;
  onViewProgress: () => void;
}

export const PublishAgentAwaitingReview: React.FC<
  PublishAgentAwaitingReviewProps
> = ({
  agentName,
  subheader,
  description,
  thumbnailSrc,
  onClose,
  onDone,
  onViewProgress,
}) => {
  return (
    <div
      className="inline-flex min-h-screen w-full flex-col items-center justify-center rounded-none bg-white sm:h-auto sm:min-h-[824px] sm:rounded-3xl dark:bg-neutral-900"
      role="dialog"
      aria-labelledby="modal-title"
    >
      <div className="relative h-[180px] w-full rounded-none bg-white sm:h-[140px] sm:rounded-t-3xl dark:bg-neutral-800">
        <div className="absolute top-[40px] left-0 flex w-full flex-col items-center justify-start px-6 sm:top-[40px]">
          <div
            id="modal-title"
            className="mb-4 text-center font-['Poppins'] text-xl leading-relaxed font-semibold text-neutral-900 sm:mb-2 sm:text-2xl dark:text-neutral-100"
          >
            Agent is awaiting review
          </div>
          <div className="max-w-[280px] text-center font-['Inter'] text-sm leading-relaxed font-normal text-slate-500 sm:max-w-none dark:text-slate-400">
            In the meantime you can check your progress on your Creator
            Dashboard page
          </div>
        </div>
        <button
          onClick={onClose}
          className="absolute top-4 right-4 flex h-[38px] w-[38px] items-center justify-center rounded-full bg-gray-100 transition-colors hover:bg-gray-200 dark:bg-neutral-700 dark:hover:bg-neutral-600"
          aria-label="Close dialog"
        >
          <IconClose
            size="default"
            className="text-neutral-600 dark:text-neutral-300"
          />
        </button>
      </div>

      <div className="flex flex-1 flex-col items-center gap-8 px-6 py-6 sm:gap-6">
        <div className="mt-4 flex w-full flex-col items-center gap-6 sm:mt-0 sm:gap-4">
          <div className="flex flex-col items-center gap-3 sm:gap-2">
            <div className="text-center font-['Geist'] text-lg leading-7 font-semibold text-neutral-800 dark:text-neutral-200">
              {agentName}
            </div>
            <div className="max-w-[280px] text-center font-['Geist'] text-base leading-normal font-normal text-neutral-600 sm:max-w-none dark:text-neutral-400">
              {subheader}
            </div>
          </div>

          <div
            className="h-[280px] w-full rounded-xl bg-neutral-200 sm:h-[350px] dark:bg-neutral-700"
            role="img"
            aria-label={
              thumbnailSrc ? "Agent thumbnail" : "Thumbnail placeholder"
            }
          >
            {thumbnailSrc && (
              <Image
                src={thumbnailSrc}
                alt="Agent thumbnail"
                width={500}
                height={350}
                className="h-full w-full rounded-xl object-cover"
              />
            )}
          </div>

          <div
            className="h-[150px] w-full overflow-y-auto font-['Geist'] text-base leading-normal font-normal text-neutral-600 sm:h-[180px] dark:text-neutral-400"
            tabIndex={0}
            role="region"
            aria-label="Agent description"
          >
            {description}
          </div>
        </div>
      </div>

      <div className="flex w-full flex-col items-center justify-center gap-4 border-t border-slate-200 p-6 sm:flex-row dark:border-slate-700">
        <Button
          onClick={onDone}
          variant="outline"
          className="h-12 w-full rounded-[59px] sm:flex-1"
        >
          Done
        </Button>
        <Button
          onClick={onViewProgress}
          variant="default"
          className="h-12 w-full rounded-[59px] bg-neutral-800 text-white hover:bg-neutral-900 sm:flex-1 dark:bg-neutral-700 dark:text-neutral-100 dark:hover:bg-neutral-600"
        >
          View progress
        </Button>
      </div>
    </div>
  );
};
