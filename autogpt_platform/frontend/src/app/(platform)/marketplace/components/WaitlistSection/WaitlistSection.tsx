"use client";

import { useState } from "react";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/__legacy__/ui/carousel";
import { WaitlistCard } from "../WaitlistCard/WaitlistCard";
import { WaitlistDetailModal } from "../WaitlistDetailModal/WaitlistDetailModal";
import type { StoreWaitlistEntry } from "@/app/api/__generated__/models/storeWaitlistEntry";
import { useWaitlistSection } from "./useWaitlistSection";

export function WaitlistSection() {
  const { waitlists, joinedWaitlistIds, isLoading, hasError, markAsJoined } =
    useWaitlistSection();
  const [selectedWaitlist, setSelectedWaitlist] =
    useState<StoreWaitlistEntry | null>(null);

  function handleOpenModal(waitlist: StoreWaitlistEntry) {
    setSelectedWaitlist(waitlist);
  }

  function handleJoinSuccess(waitlistId: string) {
    markAsJoined(waitlistId);
  }

  // Don't render if loading, error, or no waitlists
  if (isLoading || hasError || !waitlists || waitlists.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="w-full max-w-[1360px]">
        {/* Section Header */}
        <div className="mb-6">
          <h2 className="font-poppins text-2xl font-semibold text-[#282828] dark:text-neutral-200">
            Help Shape What&apos;s Next
          </h2>
          <p className="mt-2 text-base text-neutral-600 dark:text-neutral-400">
            These agents are in development. Your interest helps us prioritize
            what gets built — and we&apos;ll notify you when they&apos;re ready.
          </p>
        </div>

        {/* Mobile Carousel View */}
        <Carousel
          className="md:hidden"
          opts={{
            loop: true,
          }}
        >
          <CarouselContent>
            {waitlists.map((waitlist) => (
              <CarouselItem
                key={waitlist.waitlistId}
                className="min-w-64 max-w-71"
              >
                <WaitlistCard
                  name={waitlist.name}
                  subHeading={waitlist.subHeading}
                  description={waitlist.description}
                  imageUrl={waitlist.imageUrls[0] || null}
                  isMember={joinedWaitlistIds.has(waitlist.waitlistId)}
                  onCardClick={() => handleOpenModal(waitlist)}
                  onJoinClick={() => handleOpenModal(waitlist)}
                />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>

        {/* Desktop Grid View */}
        <div className="hidden grid-cols-1 place-items-center gap-6 md:grid md:grid-cols-2 lg:grid-cols-3">
          {waitlists.map((waitlist) => (
            <WaitlistCard
              key={waitlist.waitlistId}
              name={waitlist.name}
              subHeading={waitlist.subHeading}
              description={waitlist.description}
              imageUrl={waitlist.imageUrls[0] || null}
              isMember={joinedWaitlistIds.has(waitlist.waitlistId)}
              onCardClick={() => handleOpenModal(waitlist)}
              onJoinClick={() => handleOpenModal(waitlist)}
            />
          ))}
        </div>
      </div>

      {/* Single Modal for both viewing and joining */}
      {selectedWaitlist && (
        <WaitlistDetailModal
          waitlist={selectedWaitlist}
          isMember={joinedWaitlistIds.has(selectedWaitlist.waitlistId)}
          onClose={() => setSelectedWaitlist(null)}
          onJoinSuccess={handleJoinSuccess}
        />
      )}
    </div>
  );
}
