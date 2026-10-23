"use client";

import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";

interface BecomeACreatorProps {
  title?: string;
  description?: string;
  buttonText?: string;
  onPublish?: () => void;
}

export function BecomeACreator({
  description = "Join a community where your workflows can inspire, engage, and be installed by users around the world.",
  buttonText = "Publish your workflow",
  onPublish,
}: BecomeACreatorProps) {
  const publishTrigger = (
    <button
      // TODO(tokens): no exact role: shadow-[…] #101828, hover:shadow-[…] #101828
      className="inline-flex h-12 cursor-pointer items-center justify-center rounded-full bg-action-hover px-8 text-body-md font-medium leading-inherit text-action-foreground shadow-[0_1px_2px_rgba(16,24,40,0.1)] transition-all duration-200 hover:-translate-y-0.5 hover:bg-action hover:shadow-[0_10px_24px_-10px_rgba(16,24,40,0.4)]"
      onClick={onPublish}
    >
      {buttonText}
    </button>
  );

  return (
    <div className="relative mx-auto w-full max-w-[1360px] py-24">
      <div className="mx-auto w-full max-w-2xl px-4 text-center">
        <h2 className="mb-4 text-3xl font-semibold tracking-heading text-foreground-strong md:text-4xl">
          Build AI workflows and share{" "}
          {/* TODO(tokens): no exact role: text-violet-600 #7C3AED */}
          <span className="text-violet-600">your</span> vision
        </h2>

        <p className="mx-auto mb-8 max-w-xl text-body-md leading-relaxed text-foreground-tertiary md:text-lg">
          {description}
        </p>

        {onPublish ? (
          publishTrigger
        ) : (
          <PublishAgentModal trigger={publishTrigger} />
        )}
      </div>
    </div>
  );
}
