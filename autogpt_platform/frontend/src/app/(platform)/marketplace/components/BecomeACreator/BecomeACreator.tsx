"use client";

import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";

interface BecomeACreatorProps {
  title?: string;
  description?: string;
  buttonText?: string;
}

export function BecomeACreator({
  description = "Join a community where your workflows can inspire, engage, and be installed by users around the world.",
  buttonText = "Publish your workflow",
}: BecomeACreatorProps) {
  return (
    <div className="relative mx-auto w-full max-w-[1360px] py-24">
      <div className="mx-auto w-full max-w-2xl px-4 text-center">
        <h2 className="mb-4 text-3xl font-semibold tracking-[-0.02em] text-zinc-900 md:text-4xl">
          Build AI workflows and share{" "}
          <span className="text-violet-600">your</span> vision
        </h2>

        <p className="mx-auto mb-8 max-w-xl text-[15px] leading-relaxed text-zinc-500 md:text-lg">
          {description}
        </p>

        <PublishAgentModal
          trigger={
            <button className="inline-flex h-12 cursor-pointer items-center justify-center rounded-full bg-zinc-900 px-8 text-[15px] font-medium text-white shadow-[0_1px_2px_rgba(16,24,40,0.1)] transition-all duration-200 hover:-translate-y-0.5 hover:bg-zinc-800 hover:shadow-[0_10px_24px_-10px_rgba(16,24,40,0.4)]">
              {buttonText}
            </button>
          }
        />
      </div>
    </div>
  );
}
