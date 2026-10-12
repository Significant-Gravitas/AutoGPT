"use client";

import { Key01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { useAyrshareConnect } from "./useAyrshareConnect";

interface Props {
  className?: string;
}

// This SSO button is not a part of inputSchema - that's why we are not rendering it using Input renderer
export function AyrshareConnectButton({ className }: Props) {
  const { connect, isLoading } = useAyrshareConnect();

  // TODO :Need better UI to show user which social media accounts are connected
  return (
    <Button
      type="button"
      onClick={connect}
      disabled={isLoading}
      className={cn("h-fit w-full py-2", className)}
      loading={isLoading}
      leftIcon={<Icon icon={Key01Icon} size={16} />}
    >
      Connect Social Media Accounts
    </Button>
  );
}
