"use client";

import Image from "next/image";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { StoreWaitlistEntry } from "@/lib/autogpt-server-api/types";
import { X } from "lucide-react";

interface WaitlistDetailModalProps {
  waitlist: StoreWaitlistEntry;
  onClose: () => void;
  onJoin: () => void;
}

export function WaitlistDetailModal({
  waitlist,
  onClose,
  onJoin,
}: WaitlistDetailModalProps) {
  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <span>{waitlist.name}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Main Image */}
          {waitlist.imageUrls.length > 0 && (
            <div className="relative aspect-video w-full overflow-hidden rounded-xl">
              <Image
                src={waitlist.imageUrls[0]}
                alt={`${waitlist.name} preview`}
                fill
                className="object-cover"
              />
            </div>
          )}

          {/* Subheading */}
          <p className="text-lg font-medium text-neutral-700 dark:text-neutral-300">
            {waitlist.subHeading}
          </p>

          {/* Description */}
          <div className="prose prose-neutral dark:prose-invert max-w-none">
            <p className="whitespace-pre-wrap text-neutral-600 dark:text-neutral-400">
              {waitlist.description}
            </p>
          </div>

          {/* Video */}
          {waitlist.videoUrl && (
            <div className="space-y-2">
              <h4 className="font-medium text-neutral-800 dark:text-neutral-200">
                Video
              </h4>
              <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-neutral-100 dark:bg-neutral-800">
                <iframe
                  src={waitlist.videoUrl}
                  title={`${waitlist.name} video`}
                  className="h-full w-full"
                  allowFullScreen
                />
              </div>
            </div>
          )}

          {/* Output Demo */}
          {waitlist.agentOutputDemoUrl && (
            <div className="space-y-2">
              <h4 className="font-medium text-neutral-800 dark:text-neutral-200">
                Output Demo
              </h4>
              <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-neutral-100 dark:bg-neutral-800">
                <video
                  src={waitlist.agentOutputDemoUrl}
                  controls
                  className="h-full w-full"
                />
              </div>
            </div>
          )}

          {/* Categories */}
          {waitlist.categories.length > 0 && (
            <div className="space-y-2">
              <h4 className="font-medium text-neutral-800 dark:text-neutral-200">
                Categories
              </h4>
              <div className="flex flex-wrap gap-2">
                {waitlist.categories.map((category, index) => (
                  <span
                    key={index}
                    className="rounded-full bg-neutral-100 px-3 py-1 text-sm text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300"
                  >
                    {category}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Join Button */}
          <Button
            onClick={onJoin}
            className="w-full rounded-full bg-neutral-800 text-white hover:bg-neutral-700 dark:bg-neutral-700 dark:hover:bg-neutral-600"
          >
            Join waitlist
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
