"use client";

import { PlusIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onCreate: () => void;
}

export function APIKeysHeader({ onCreate }: Props) {
  return (
    <div className="flex flex-col items-start gap-4 pb-6 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="h4" as="h1" className="leading-[28px] text-textBlack">
          AutoGPT API Keys
        </Text>
        <Text variant="body" className="mt-4 max-w-[600px] text-zinc-700">
          Manage API keys that let external tools access your AutoGPT account.
        </Text>
      </div>

      <Button
        variant="primary"
        size="small"
        leftIcon={<PlusIcon size={16} />}
        onClick={onCreate}
        className="sm:hidden"
      >
        Create Key
      </Button>
      <Button
        variant="primary"
        size="large"
        leftIcon={<PlusIcon size={20} />}
        onClick={onCreate}
        className="hidden sm:inline-flex"
      >
        Create Key
      </Button>
    </div>
  );
}
