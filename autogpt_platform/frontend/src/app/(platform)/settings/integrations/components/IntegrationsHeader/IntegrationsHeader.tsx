"use client";

import { PlusIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onConnect: () => void;
}

export function IntegrationsHeader({ onConnect }: Props) {
  return (
    <div className="flex flex-col items-start gap-4 pb-6 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="h4" as="h1" className="leading-[28px] text-[#1F1F20]">
          Third Party Integrations
        </Text>
        <Text variant="body" className="mt-4 max-w-[600px] text-[#505057]">
          Manage the 3rd party accounts you&apos;ve connected to AutoGPT. These
          are services that can be used by your agents — like Gmail for sending
          emails, GitHub for code, or Figma for designs.
        </Text>
      </div>

      <Button
        variant="primary"
        size="small"
        leftIcon={<PlusIcon size={16} />}
        onClick={onConnect}
        className="sm:hidden"
      >
        Connect Service
      </Button>
      <Button
        variant="primary"
        size="large"
        leftIcon={<PlusIcon size={20} />}
        onClick={onConnect}
        className="hidden sm:inline-flex"
      >
        Connect Service
      </Button>
    </div>
  );
}
