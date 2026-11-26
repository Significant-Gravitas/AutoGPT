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
      className="inline-flex h-12 cursor-pointer items-center justify-center rounded-pill bg-action-hover px-8 text-body-md leading-inherit font-medium text-action-foreground shadow-sm shadow-scrim/10 transition-all duration-200 hover:-translate-y-0.5 hover:bg-action hover:shadow-drop hover:shadow-scrim/40"
      onClick={onPublish}
    >
      {buttonText}
    </button>
  );

  return (
    <div className="relative mx-auto w-full max-w-[1360px] py-24">
      <div className="mx-auto w-full max-w-2xl px-4 text-center">
        <h2 className="mb-4 text-heading font-semibold tracking-heading text-foreground-strong md:text-title">
          Build AI workflows and share <span className="text-brand">your</span>{" "}
          vision
        </h2>

        <p className="mx-auto mb-8 max-w-xl text-body-md leading-relaxed text-foreground-tertiary md:text-lead">
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
