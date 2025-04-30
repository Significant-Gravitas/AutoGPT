"use client";

import * as React from "react";
import { IconClose } from "../ui/icons";
import Image from "next/image";
import { Button } from "../agptui/Button";
import { X } from "lucide-react";

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
      role="dialog"
      aria-labelledby="modal-title"
      className="m-auto flex h-fit w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg dark:bg-gray-800"
    >
      {/* Top  */}
      <div className="relative items-center justify-center border-b border-slate-200 pb-4 pt-12 dark:border-slate-700 md:flex md:h-28">
        <div className="absolute right-4 top-4">
          <Button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-transparent p-0 transition-colors hover:bg-gray-200"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="px-4 text-center">
          <h3 className="font-poppins text-lg font-semibold text-neutral-900 md:text-2xl">
            Agent is awaiting review
          </h3>
          <p className="hidden font-sans text-sm font-normal text-neutral-600 sm:flex">
            In the meantime you can check your progress on your Creator
            Dashboard page
          </p>
        </div>
      </div>

      <div className="h-[50vh] flex-grow space-y-5 overflow-y-auto p-4 md:h-[38rem] md:p-6">
        <div className="mt-4 flex w-full flex-col items-center gap-6 sm:mt-0 sm:gap-4">
          {/* Heading */}
          <div className="flex flex-col items-center gap-3 sm:gap-2">
            <div className="text-center font-sans text-lg font-semibold leading-7 text-neutral-800 dark:text-neutral-200">
              {agentName}
            </div>
            <div className="max-w-[280px] text-center font-sans text-base font-normal leading-normal text-neutral-600 dark:text-neutral-400 sm:max-w-none">
              {subheader}
            </div>
          </div>

          {/* Image */}
          <div
            className="h-[280px] w-full rounded-xl bg-neutral-200 dark:bg-neutral-700 sm:h-[350px]"
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

          {/* Description */}
          <div
            className="w-full whitespace-pre-line font-sans text-base font-normal text-neutral-600 dark:text-neutral-400"
            tabIndex={0}
            role="region"
            aria-label="Agent description"
          >
            {description}
          </div>
        </div>
      </div>

      <div className="flex w-full flex-col items-center justify-center gap-4 border-t border-slate-200 p-6 dark:border-slate-700 sm:flex-row">
        <Button
          onClick={onDone}
          className="flex h-12 w-full items-center justify-center rounded-[59px] sm:flex-1"
        >
          Done
        </Button>
        <Button
          onClick={onViewProgress}
          className="flex h-12 w-full items-center justify-center rounded-[59px] bg-neutral-800 text-white hover:bg-neutral-900 dark:bg-neutral-700 dark:text-neutral-100 dark:hover:bg-neutral-600 sm:flex-1"
        >
          View progress
        </Button>
      </div>
    </div>
  );
};
