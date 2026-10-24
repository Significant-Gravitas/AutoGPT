"use client";

import { PlusSignIcon } from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onCreate: () => void;
}

export function KeylessAccessHeader({ onCreate }: Props) {
  return (
    <div className="flex flex-col items-start gap-4 pb-6 pl-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="h4" as="h1" className="leading-[28px] text-textBlack">
          Keyless API Access
        </Text>
        <Text variant="body" className="mt-4 max-w-[600px] text-zinc-700">
          Let a CI job, server or website call the AutoGPT API without an API
          key, using the identity its platform already gives it.
        </Text>
      </div>

      <Button
        variant="primary"
        size="small"
        leftIcon={<Icon icon={PlusSignIcon} size={16} />}
        onClick={onCreate}
        className="sm:hidden"
      >
        Add workload
      </Button>
      <Button
        variant="primary"
        size="large"
        leftIcon={<Icon icon={PlusSignIcon} size={20} />}
        onClick={onCreate}
        className="hidden sm:inline-flex"
      >
        Add workload
      </Button>
    </div>
  );
}
