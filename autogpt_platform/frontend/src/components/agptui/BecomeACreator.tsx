"use client";

import * as React from "react";
import { PublishAgentPopout } from "./composite/PublishAgentPopout";
import { Button } from "@/components/ui/button";
interface BecomeACreatorProps {
  title?: string;
  description?: string;
  buttonText?: string;
  onButtonClick?: () => void;
}

export const BecomeACreator: React.FC<BecomeACreatorProps> = ({
  title = "Become a creator",
  description = "Join a community where your AI creations can inspire, engage, and be downloaded by users around the world.",
  buttonText = "Upload your agent",
  onButtonClick,
}) => {
  const handleButtonClick = () => {
    onButtonClick?.();
    console.log("Become A Creator clicked");
  };

  return (
    <div className="relative mx-auto h-auto min-h-[300px] w-full max-w-[1360px] md:min-h-[400px] lg:h-[459px]">
      {/* Top border */}
      <div className="left-0 top-0 h-px w-full bg-gray-200 dark:bg-gray-700" />

      {/* Title */}
      <h2 className="underline-from-font decoration-skip-ink-none mt-[25px] text-left font-poppins text-[18px] font-semibold leading-[28px] text-neutral-800 dark:text-neutral-200">
        {title}
      </h2>

      {/* Content Container */}
      <div className="m-auto w-full max-w-[900px] px-4 py-16 text-center md:px-6 lg:px-0">
        <h2 className="underline-from-font decoration-skip-ink-none mb-6 text-center font-poppins text-[48px] font-semibold leading-[54px] tracking-[-0.012em] text-neutral-950 dark:text-neutral-50 md:mb-8 lg:mb-12">
          Build AI agents and share
          <br />
          <span className="text-violet-600 dark:text-violet-400">
            your
          </span>{" "}
          vision
        </h2>

        <p className="font-geist mx-auto mb-8 max-w-[90%] text-lg font-normal leading-relaxed text-neutral-700 dark:text-neutral-300 md:mb-10 md:text-xl md:leading-loose lg:mb-14 lg:text-2xl">
          {description}
        </p>

        <PublishAgentPopout
          trigger={
            <Button
              onClick={handleButtonClick}
            >
              {buttonText}
            </Button>
          }
        />
      </div>
    </div>
  );
};
