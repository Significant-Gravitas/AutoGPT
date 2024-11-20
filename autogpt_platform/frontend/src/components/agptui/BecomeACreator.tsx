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
    <div className="relative h-auto min-h-[300px] md:min-h-[400px] lg:h-[459px] w-full max-w-[1360px] mx-auto px-4 md:px-6 lg:px-8">
      {/* Top border */}
      <div className="absolute left-0 top-0 h-px w-full bg-gray-200" />
      
      {/* Title */}
      <div className="absolute left-4 md:left-6 lg:left-8 top-[26px] text-base md:text-lg font-semibold font-poppins leading-7 text-neutral-800">
        {title}
      </div>

      {/* Content Container - Centered */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-[900px] text-center px-4 md:px-6 lg:px-0 pt-[40px]">
        {/* Heading with highlighted word */}
        <h2 className="text-3xl md:text-4xl lg:text-5xl font-semibold font-poppins leading-tight md:leading-[1.2] lg:leading-[54px] text-neutral-950 mb-6 md:mb-8 lg:mb-12">
          Build AI agents and share{' '}
          <span className="text-violet-600">your</span>
          {' '}vision
        </h2>

        {/* Description */}
        <p className="font-geist text-lg md:text-xl lg:text-2xl font-normal leading-relaxed md:leading-loose text-neutral-700 mb-8 md:mb-10 lg:mb-14 max-w-[90%] mx-auto">
          {description}
        </p>

        {/* Button */}
        <button 
          onClick={handleButtonClick}
          className="inline-flex h-[48px] md:h-[56px] lg:h-[68px] cursor-pointer items-center justify-center rounded-[38px] bg-neutral-800 px-4 md:px-5 lg:px-6 py-3 md:py-4 lg:py-5 hover:bg-neutral-700 transition-colors"
        >
          <span className="font-poppins text-base md:text-lg lg:text-xl font-medium leading-normal md:leading-relaxed lg:leading-7 text-neutral-50 whitespace-nowrap">
            {buttonText}
          </span>
        </button>
      </div>
    </div>
  );
};
