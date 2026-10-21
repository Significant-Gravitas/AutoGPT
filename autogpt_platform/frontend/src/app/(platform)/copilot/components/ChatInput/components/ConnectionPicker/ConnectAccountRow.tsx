"use client";

import { Loading03Icon, PlusSignIcon } from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
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
    <Button
      type="button"
      variant="ghost"
      size="small"
      aria-label="Connect a ChatGPT subscription"
      disabled={isConnecting}
      onClick={onConnect}
      className="h-auto min-h-14 w-full justify-start gap-3 whitespace-normal rounded-lg px-3 py-2 text-left"
    >
      <IntegrationLogo provider="openai" size={20} />
      <span className="min-w-0 flex-1 text-sm font-medium text-zinc-900">
        <span className="block">ChatGPT subscription</span>
        <span className="mt-1 block text-[11px] font-normal text-zinc-600">
          Connect your existing account
        </span>
      </span>
      <span
        role={isConnecting ? "status" : undefined}
        aria-label={isConnecting ? "Connecting ChatGPT" : undefined}
        className="flex size-7 flex-none items-center justify-center rounded-lg border border-zinc-200 bg-white text-zinc-500"
      >
        <Icon
          icon={isConnecting ? Loading03Icon : PlusSignIcon}
          size={14}
          aria-hidden
          className={isConnecting ? "animate-spin" : undefined}
        />
      </span>
    </Button>
  );
}
