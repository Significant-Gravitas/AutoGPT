"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
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
        <Text
          variant="lead-semibold"
          as="h2"
          tone="primary"
          className="mb-4 text-3xl tracking-[-0.02em] md:text-4xl"
        >
          Build AI workflows and share{" "}
          <span className="text-purple-600">your</span> vision
        </Text>

        <Text
          variant="large"
          tone="muted"
          className="mx-auto mb-8 max-w-xl text-[15px] leading-relaxed md:text-lg"
        >
          {description}
        </Text>

        <PublishAgentModal
          trigger={
            <Button
              variant="primary"
              className="h-12 cursor-pointer border-0 bg-zinc-900 px-8 text-[15px] shadow-[0_1px_2px_rgba(16,24,40,0.1)] transition-all duration-200 hover:-translate-y-0.5 hover:bg-zinc-800 hover:shadow-[0_10px_24px_-10px_rgba(16,24,40,0.4)]"
            >
              {buttonText}
            </Button>
          }
        />
      </div>
    </div>
  );
}
