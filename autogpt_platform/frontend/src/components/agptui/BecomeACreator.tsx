"use client";

import * as React from "react";
import { Button } from "./Button";

interface BecomeACreatorProps {
  title: string;
  heading: string;
  description: string;
  buttonText: string;
}

export const BecomeACreator: React.FC<BecomeACreatorProps> = ({
  title = "Become a creator",
  heading = "Build AI agents and share your vision",
  description = "Join a community where your AI creations can inspire, engage,\nand be downloaded by users around the world.",
  buttonText = "Upload your agent",
}) => {
  const handleButtonClick = () => {
    console.log("Upload agent clicked");
  };

  return (
    <div className="relative mx-auto h-auto min-h-[300px] w-full max-w-[1360px] px-4 md:min-h-[400px] md:px-6 lg:h-[459px] lg:px-8">
      {/* Top border */}
      <div className="absolute left-0 top-0 h-px w-full bg-gray-200" />

      {/* Title */}
      <div className="font-poppins absolute left-4 top-[26px] text-base font-semibold leading-7 text-neutral-800 md:left-6 md:text-lg lg:left-8">
        {title}
      </div>

      {/* Content Container - Centered */}
      <div className="absolute left-1/2 top-1/2 w-full max-w-[900px] -translate-x-1/2 -translate-y-1/2 px-4 pt-[40px] text-center md:px-6 lg:px-0">
        {/* Heading with highlighted word */}
        <h2 className="font-poppins mb-6 text-3xl font-semibold leading-tight text-neutral-950 md:mb-8 md:text-4xl md:leading-[1.2] lg:mb-12 lg:text-5xl lg:leading-[54px]">
          Build AI agents and share{" "}
          <span className="text-violet-600">your</span> vision
        </h2>

        {/* Description */}
        <p className="font-geist mx-auto mb-8 max-w-[90%] text-lg font-normal leading-relaxed text-neutral-700 md:mb-10 md:text-xl md:leading-loose lg:mb-14 lg:text-2xl">
          {description}
        </p>

        {/* Button */}
        <button
          onClick={handleButtonClick}
          className="inline-flex h-[48px] cursor-pointer items-center justify-center rounded-[38px] bg-neutral-800 px-4 py-3 transition-colors hover:bg-neutral-700 md:h-[56px] md:px-5 md:py-4 lg:h-[68px] lg:px-6 lg:py-5"
        >
          <span className="font-poppins whitespace-nowrap text-base font-medium leading-normal text-neutral-50 md:text-lg md:leading-relaxed lg:text-xl lg:leading-7">
            {buttonText}
          </span>
        </button>
      </div>
    </div>
  );
};
