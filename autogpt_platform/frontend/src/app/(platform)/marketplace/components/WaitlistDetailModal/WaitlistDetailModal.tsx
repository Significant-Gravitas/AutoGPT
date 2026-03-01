"use client";

import { useState } from "react";
import Image from "next/image";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Input } from "@/components/atoms/Input/Input";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/__legacy__/ui/carousel";
import type { StoreWaitlistEntry } from "@/app/api/__generated__/models/storeWaitlistEntry";
import { Check, Play } from "@phosphor-icons/react";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2AddSelfToTheAgentWaitlist } from "@/app/api/__generated__/endpoints/store/store";

interface MediaItem {
  type: "image" | "video";
  url: string;
  label?: string;
}

// Extract YouTube video ID from various URL formats
function getYouTubeVideoId(url: string): string | null {
  const regExp =
    /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
  const match = url.match(regExp);
  return match && match[7].length === 11 ? match[7] : null;
}

// Validate video URL for security
function isValidVideoUrl(url: string): boolean {
  if (url.startsWith("data:video")) {
    return true;
  }
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
  const validUrl = /^(https?:\/\/)/i;
  const cleanedUrl = url.split("?")[0];
  return (
    (validUrl.test(url) && videoExtensions.test(cleanedUrl)) ||
    youtubeRegex.test(url)
  );
}

// Video player with YouTube embed support
function VideoPlayer({
  url,
  autoPlay = false,
  className = "",
}: {
  url: string;
  autoPlay?: boolean;
  className?: string;
}) {
  const youtubeId = getYouTubeVideoId(url);

  if (youtubeId) {
    return (
      <iframe
        src={`https://www.youtube.com/embed/${youtubeId}${autoPlay ? "?autoplay=1" : ""}`}
        title="YouTube video player"
        className={className}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        sandbox="allow-same-origin allow-scripts allow-presentation"
        allowFullScreen
      />
    );
  }

  if (!isValidVideoUrl(url)) {
    return (
      <div
        className={`flex items-center justify-center bg-zinc-800 ${className}`}
      >
        <span className="text-sm text-zinc-400">Invalid video URL</span>
      </div>
    );
  }

  return <video src={url} controls autoPlay={autoPlay} className={className} />;
}

function MediaCarousel({ waitlist }: { waitlist: StoreWaitlistEntry }) {
  const [activeVideo, setActiveVideo] = useState<string | null>(null);

  // Build media items array: videos first, then images
  const mediaItems: MediaItem[] = [
    ...(waitlist.videoUrl
      ? [{ type: "video" as const, url: waitlist.videoUrl, label: "Video" }]
      : []),
    ...(waitlist.agentOutputDemoUrl
      ? [
          {
            type: "video" as const,
            url: waitlist.agentOutputDemoUrl,
            label: "Demo",
          },
        ]
      : []),
    ...waitlist.imageUrls.map((url) => ({ type: "image" as const, url })),
  ];

  if (mediaItems.length === 0) return null;

  // Single item - no carousel needed
  if (mediaItems.length === 1) {
    const item = mediaItems[0];
    return (
      <div className="relative aspect-[350/196] w-full overflow-hidden rounded-large">
        {item.type === "image" ? (
          <Image
            src={item.url}
            alt={`${waitlist.name} preview`}
            fill
            className="object-cover"
          />
        ) : (
          <VideoPlayer url={item.url} className="h-full w-full object-cover" />
        )}
      </div>
    );
  }

  // Multiple items - use carousel
  return (
    <Carousel className="w-full">
      <CarouselContent>
        {mediaItems.map((item, index) => (
          <CarouselItem key={index}>
            <div className="relative aspect-[350/196] w-full overflow-hidden rounded-large">
              {item.type === "image" ? (
                <Image
                  src={item.url}
                  alt={`${waitlist.name} preview ${index + 1}`}
                  fill
                  className="object-cover"
                />
              ) : activeVideo === item.url ? (
                <VideoPlayer
                  url={item.url}
                  autoPlay
                  className="h-full w-full object-cover"
                />
              ) : (
                <button
                  onClick={() => setActiveVideo(item.url)}
                  className="group relative h-full w-full bg-zinc-900"
                >
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/90 transition-transform group-hover:scale-110">
                      <Play size={32} weight="fill" className="text-zinc-800" />
                    </div>
                  </div>
                  <span className="absolute bottom-3 left-3 text-sm text-white">
                    {item.label}
                  </span>
                </button>
              )}
            </div>
          </CarouselItem>
        ))}
      </CarouselContent>
      <CarouselPrevious className="left-2 top-1/2 -translate-y-1/2" />
      <CarouselNext className="right-2 top-1/2 -translate-y-1/2" />
    </Carousel>
  );
}

interface WaitlistDetailModalProps {
  waitlist: StoreWaitlistEntry;
  isMember?: boolean;
  onClose: () => void;
  onJoinSuccess?: (waitlistId: string) => void;
}

export function WaitlistDetailModal({
  waitlist,
  isMember = false,
  onClose,
  onJoinSuccess,
}: WaitlistDetailModalProps) {
  const { user } = useSupabaseStore();
  const [email, setEmail] = useState("");
  const [success, setSuccess] = useState(false);
  const { toast } = useToast();
  const joinWaitlistMutation = usePostV2AddSelfToTheAgentWaitlist();

  function handleJoin() {
    joinWaitlistMutation.mutate(
      {
        waitlistId: waitlist.waitlistId,
        data: { email: user ? undefined : email },
      },
      {
        onSuccess: (response) => {
          if (response.status === 200) {
            setSuccess(true);
            toast({
              title: "You're on the waitlist!",
              description: `We'll notify you when ${waitlist.name} goes live.`,
            });
            onJoinSuccess?.(waitlist.waitlistId);
          } else {
            toast({
              variant: "destructive",
              title: "Error",
              description: "Failed to join waitlist. Please try again.",
            });
          }
        },
        onError: () => {
          toast({
            variant: "destructive",
            title: "Error",
            description: "Failed to join waitlist. Please try again.",
          });
        },
      },
    );
  }

  // Success state
  if (success) {
    return (
      <Dialog
        title=""
        controlled={{
          isOpen: true,
          set: async (open) => {
            if (!open) onClose();
          },
        }}
        onClose={onClose}
        styling={{ maxWidth: "500px" }}
      >
        <Dialog.Content>
          <div className="flex flex-col items-center justify-center py-4 text-center">
            {/* Party emoji */}
            <span className="mb-2 text-5xl">🎉</span>

            {/* Title */}
            <h2 className="mb-2 font-poppins text-[22px] font-medium leading-7 text-zinc-900 dark:text-zinc-100">
              You&apos;re on the waitlist
            </h2>

            {/* Subtitle */}
            <p className="text-base leading-[26px] text-zinc-600 dark:text-zinc-400">
              Thanks for helping us prioritize which agents to build next.
              We&apos;ll notify you when this agent goes live in the
              marketplace.
            </p>
          </div>

          {/* Close button */}
          <Dialog.Footer className="flex justify-center pb-2 pt-4">
            <Button
              variant="secondary"
              onClick={onClose}
              className="rounded-full border border-zinc-700 bg-white px-4 py-3 text-zinc-900 hover:bg-zinc-100 dark:border-zinc-500 dark:bg-zinc-800 dark:text-zinc-100 dark:hover:bg-zinc-700"
            >
              Close
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    );
  }

  // Main modal - handles both member and non-member states
  return (
    <Dialog
      title="Join the waitlist"
      controlled={{
        isOpen: true,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "500px" }}
    >
      <Dialog.Content>
        {/* Subtitle */}
        <p className="mb-6 text-center text-base text-zinc-600 dark:text-zinc-400">
          Help us decide what to build next — and get notified when this agent
          is ready
        </p>

        {/* Media Carousel */}
        <MediaCarousel waitlist={waitlist} />

        {/* Agent Name */}
        <h3 className="mt-4 font-poppins text-[22px] font-medium leading-7 text-zinc-800 dark:text-zinc-100">
          {waitlist.name}
        </h3>

        {/* Agent Description */}
        <p className="mt-2 line-clamp-5 text-sm leading-[22px] text-zinc-500 dark:text-zinc-400">
          {waitlist.description}
        </p>

        {/* Email input for non-logged-in users who haven't joined */}
        {!isMember && !user && (
          <div className="mt-4 pr-1">
            <Input
              id="email"
              label="Email address"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
        )}

        {/* Footer buttons */}
        <Dialog.Footer className="sticky bottom-0 mt-6 flex justify-center gap-3 bg-white pb-2 pt-4 dark:bg-zinc-900">
          {isMember ? (
            <Button
              disabled
              className="rounded-full bg-green-600 px-4 py-3 text-white hover:bg-green-600 dark:bg-green-700 dark:hover:bg-green-700"
            >
              <Check size={16} className="mr-2" />
              You&apos;re on the waitlist
            </Button>
          ) : (
            <>
              <Button
                onClick={handleJoin}
                loading={joinWaitlistMutation.isPending}
                disabled={!user && !email}
                className="rounded-full bg-zinc-800 px-4 py-3 text-white hover:bg-zinc-700 dark:bg-zinc-700 dark:hover:bg-zinc-600"
              >
                Join waitlist
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={onClose}
                className="rounded-full bg-zinc-200 px-4 py-3 text-zinc-900 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-600"
              >
                Not now
              </Button>
            </>
          )}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
