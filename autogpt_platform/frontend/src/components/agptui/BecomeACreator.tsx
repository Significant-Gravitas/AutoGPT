"use client";

import * as React from "react";
import { PublishAgentPopout } from "./composite/PublishAgentPopout";
import AutogptButton from "./AutogptButton";
interface BecomeACreatorProps {
  title?: string;
  buttonText?: string;
  onButtonClick?: () => void;
}

export const BecomeACreator: React.FC<BecomeACreatorProps> = ({
  title = "Become a creator",
  buttonText = "Upload your agent",
  onButtonClick,
}) => {
  const handleButtonClick = () => {
    onButtonClick?.();
  };

  return (
    <div className="w-full space-y-18 sm:mb-36 md:mb-72">
      {/* Title */}
      <h2 className="font-poppins text-base font-medium text-zinc-500">
        {title}
      </h2>

      {/* Content Container */}
      <div className="flex flex-col items-center justify-center">
        <h2 className="mb-9 text-center font-poppins text-3xl font-semibold leading-[3.5rem] text-neutral-950 md:text-[2.75rem]">
          Build AI agents and share
          <span className="text-violet-600"> your </span>
          vision
        </h2>

        <p className="mb-12 text-center font-sans text-lg font-normal text-zinc-600">
          Join a community where your AI creations can inspire, engage, <br />{" "}
          and be downloaded by users around the world.
        </p>

        <PublishAgentPopout
          trigger={
            <AutogptButton onClick={handleButtonClick}>
              {buttonText}
            </AutogptButton>
          }
        />
      </div>
    </div>
  );
};
