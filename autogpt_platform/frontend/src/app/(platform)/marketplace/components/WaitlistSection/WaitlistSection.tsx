"use client";

import { useState } from "react";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/__legacy__/ui/carousel";
import { WaitlistCard } from "../WaitlistCard/WaitlistCard";
import { WaitlistDetailModal } from "../WaitlistDetailModal/WaitlistDetailModal";
import { JoinWaitlistModal } from "../JoinWaitlistModal/JoinWaitlistModal";
import { StoreWaitlistEntry } from "@/lib/autogpt-server-api/types";
import { useWaitlistSection } from "./useWaitlistSection";

export function WaitlistSection() {
  const { waitlists, isLoading, hasError } = useWaitlistSection();
  const [selectedWaitlist, setSelectedWaitlist] =
    useState<StoreWaitlistEntry | null>(null);
  const [joiningWaitlist, setJoiningWaitlist] =
    useState<StoreWaitlistEntry | null>(null);

  function handleCardClick(waitlist: StoreWaitlistEntry) {
    setSelectedWaitlist(waitlist);
  }

  function handleJoinClick(waitlist: StoreWaitlistEntry) {
    setJoiningWaitlist(waitlist);
  }

  function handleJoinFromDetail() {
    if (selectedWaitlist) {
      setJoiningWaitlist(selectedWaitlist);
      setSelectedWaitlist(null);
    }
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
                key={waitlist.waitlist_id}
                className="min-w-64 max-w-71"
              >
                <WaitlistCard
                  name={waitlist.name}
                  subHeading={waitlist.subHeading}
                  description={waitlist.description}
                  imageUrl={waitlist.imageUrls[0] || null}
                  onCardClick={() => handleCardClick(waitlist)}
                  onJoinClick={() => handleJoinClick(waitlist)}
                />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>

        {/* Desktop Grid View */}
        <div className="hidden grid-cols-1 place-items-center gap-6 md:grid md:grid-cols-2 lg:grid-cols-3">
          {waitlists.map((waitlist) => (
            <WaitlistCard
              key={waitlist.waitlist_id}
              name={waitlist.name}
              subHeading={waitlist.subHeading}
              description={waitlist.description}
              imageUrl={waitlist.imageUrls[0] || null}
              onCardClick={() => handleCardClick(waitlist)}
              onJoinClick={() => handleJoinClick(waitlist)}
            />
          ))}
        </div>
      </div>

      {/* Detail Modal */}
      {selectedWaitlist && (
        <WaitlistDetailModal
          waitlist={selectedWaitlist}
          onClose={() => setSelectedWaitlist(null)}
          onJoin={handleJoinFromDetail}
        />
      )}

      {/* Join Modal */}
      {joiningWaitlist && (
        <JoinWaitlistModal
          waitlist={joiningWaitlist}
          onClose={() => setJoiningWaitlist(null)}
        />
      )}
    </div>
  );
}
