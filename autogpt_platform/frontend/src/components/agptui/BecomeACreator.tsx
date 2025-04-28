"use client";

import * as React from "react";
import { PublishAgentPopout } from "./composite/PublishAgentPopout";
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
  };

  return (
    <div className="mb-18 w-full sm:mb-36 md:mb-72">
      {/* Title */}
      <h2 className="mb-18 font-poppins text-lg font-semibold text-neutral-800">
        {title}
      </h2>

      {/* Content Container */}
      <div className="flex flex-col items-center justify-center">
        <h2 className="mb-9 text-center font-poppins text-3xl font-semibold leading-[3rem] text-neutral-950 md:text-5xl">
          Build AI agents and share
          <br />
          your vision
        </h2>

        <p className="mb-12 text-center font-sans text-lg font-normal text-neutral-700 md:text-2xl">
          {description}
        </p>

        <PublishAgentPopout
          trigger={
            <button
              onClick={handleButtonClick}
              className="inline-flex h-[48px] cursor-pointer items-center justify-center rounded-[38px] bg-neutral-800 px-8 py-3 transition-colors hover:bg-neutral-700 dark:bg-neutral-700 dark:hover:bg-neutral-600 md:h-[56px] md:px-10 md:py-4 lg:h-[68px] lg:px-12 lg:py-5"
            >
              <span className="whitespace-nowrap font-poppins text-base font-medium leading-normal text-neutral-50 md:text-lg md:leading-relaxed lg:text-xl lg:leading-7">
                {buttonText}
              </span>
            </button>
          }
        />
      </div>
    </div>
  );
};
