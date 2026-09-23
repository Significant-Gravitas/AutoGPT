"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  onConnect: () => void;
  withTitle?: boolean;
}

export function IntegrationsHeader({ onConnect, withTitle = true }: Props) {
  return (
    <div className="flex flex-col items-start gap-4 pb-6 pl-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex min-w-0 flex-col">
        {withTitle && (
          <Text variant="h4" as="h1" className="leading-[28px] text-[#1F1F20]">
            Integrations
          </Text>
        )}
        <Text variant="body" className="mt-4 max-w-[600px] text-[#505057]">
          Connect AI subscriptions to power your agents, and third-party tools
          for them to use.
        </Text>
      </div>

      <Button
        variant="primary"
        size="small"
        leftIcon={<Icon icon={PlusSignIcon} size={16} />}
        onClick={onConnect}
        className="sm:hidden"
      >
        Connect Service
      </Button>
      <Button
        variant="primary"
        size="large"
        leftIcon={<Icon icon={PlusSignIcon} size={20} />}
        onClick={onConnect}
        className="hidden sm:inline-flex"
      >
        Connect Service
      </Button>
    </div>
  );
}
