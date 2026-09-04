"use client";

import { PlusSignIcon } from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";

interface Props {
  onConnect: () => void;
  isConnecting: boolean;
}

/**
 * The way in for a subscription the user has not linked yet.
 *
 * Without it the picker can only list what already exists, so a user who
 * has never connected ChatGPT sees a control about connections that offers
 * no way to make one, and has to discover Settings to act on it.
 */
export function ConnectAccountRow({ onConnect, isConnecting }: Props) {
  return (
    <div className="flex items-center gap-3 px-3 py-2">
      <IntegrationLogo provider="openai" size={20} />
      {/* The row sits under "Add a connection" and beside the connections it
          would join, so what it is for is already said. A sentence of its own
          would only repeat the section above it. */}
      <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-900">
        ChatGPT subscription
      </span>
      <Button
        variant="primary"
        size="small"
        aria-label="Connect a ChatGPT subscription"
        // Not the Button's own `loading`: it keeps the children beside its
        // spinner, and drops the props that carry this button's only name.
        disabled={isConnecting}
        onClick={onConnect}
        className="h-8 w-8 min-w-0 flex-none p-0"
      >
        {isConnecting ? (
          <LoadingSpinner size="small" />
        ) : (
          <Icon icon={PlusSignIcon} size={16} aria-hidden />
        )}
      </Button>
    </div>
  );
}
