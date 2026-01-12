"use client";

import Image from "next/image";
import { Button } from "@/components/atoms/Button/Button";
import { Check } from "@phosphor-icons/react";

interface WaitlistCardProps {
  name: string;
  subHeading: string;
  description: string;
  imageUrl: string | null;
  isMember?: boolean;
  onCardClick: () => void;
  onJoinClick: (e: React.MouseEvent) => void;
}

export function WaitlistCard({
  name,
  subHeading,
  description,
  imageUrl,
  isMember = false,
  onCardClick,
  onJoinClick,
}: WaitlistCardProps) {
  function handleJoinClick(e: React.MouseEvent) {
    e.stopPropagation();
    onJoinClick(e);
  }

  return (
    <div
      className="flex h-[24rem] w-full max-w-md cursor-pointer flex-col items-start rounded-3xl bg-white transition-all duration-300 hover:shadow-lg dark:bg-zinc-900 dark:hover:shadow-gray-700"
      onClick={onCardClick}
      data-testid="waitlist-card"
      role="button"
      tabIndex={0}
      aria-label={`${name} waitlist card`}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          onCardClick();
        }
      }}
    >
      {/* Image Section */}
      <div className="relative aspect-[2/1.2] w-full overflow-hidden rounded-large md:aspect-[2.17/1]">
        {imageUrl ? (
          <Image
            src={imageUrl}
            alt={`${name} preview image`}
            fill
            className="object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-neutral-200 to-neutral-300 dark:from-neutral-700 dark:to-neutral-800">
            <span className="text-4xl font-bold text-neutral-400 dark:text-neutral-500">
              {name.charAt(0)}
            </span>
          </div>
        )}
      </div>

      <div className="mt-3 flex w-full flex-1 flex-col px-4">
        {/* Name and Subheading */}
        <div className="flex w-full flex-col">
          <h3 className="line-clamp-1 font-poppins text-xl font-semibold text-[#272727] dark:text-neutral-100">
            {name}
          </h3>
          <p className="mt-1 line-clamp-1 text-sm text-neutral-500 dark:text-neutral-400">
            {subHeading}
          </p>
        </div>

        {/* Description */}
        <div className="mt-2 flex w-full flex-col">
          <p className="line-clamp-5 text-sm font-normal leading-relaxed text-neutral-600 dark:text-neutral-400">
            {description}
          </p>
        </div>

        <div className="flex-grow" />

        {/* Join Waitlist Button */}
        <div className="mt-4 w-full pb-4">
          {isMember ? (
            <Button
              disabled
              className="w-full rounded-full bg-green-600 text-white hover:bg-green-600 dark:bg-green-700 dark:hover:bg-green-700"
            >
              <Check className="mr-2" size={16} weight="bold" />
              On the waitlist
            </Button>
          ) : (
            <Button
              onClick={handleJoinClick}
              className="w-full rounded-full bg-zinc-800 text-white hover:bg-zinc-700 dark:bg-zinc-700 dark:hover:bg-zinc-600"
            >
              Join waitlist
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
