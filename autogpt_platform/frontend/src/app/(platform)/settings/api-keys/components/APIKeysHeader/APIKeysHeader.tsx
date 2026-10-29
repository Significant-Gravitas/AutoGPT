"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const API_DOCS_URL =
  "https://agpt.co/docs/platform/api-and-integrations/api-guide";

interface Props {
  onCreate: () => void;
}

export function APIKeysHeader({ onCreate }: Props) {
  return (
    <div className="flex flex-col items-start gap-4 pb-6 pl-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="h4" as="h1" className="leading-7 text-textBlack">
          AutoGPT API Keys
        </Text>
        <Text
          variant="body"
          className="mt-4 max-w-[600px] text-foreground-body"
        >
          Manage API keys that let external tools access your AutoGPT account.
          See the{" "}
          <Link href={API_DOCS_URL} isExternal variant="secondary">
            API docs
          </Link>{" "}
          to get started.
        </Text>
      </div>

      <Button
        variant="primary"
        size="small"
        leftIcon={<Icon icon={PlusSignIcon} size={16} />}
        onClick={onCreate}
        className="sm:hidden"
      >
        Create Key
      </Button>
      <Button
        variant="primary"
        size="large"
        leftIcon={<Icon icon={PlusSignIcon} size={20} />}
        onClick={onCreate}
        className="hidden sm:inline-flex"
      >
        Create Key
      </Button>
    </div>
  );
}
