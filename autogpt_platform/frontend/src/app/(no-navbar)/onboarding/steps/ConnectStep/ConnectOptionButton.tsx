"use client";

import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import { useQueryClient } from "@tanstack/react-query";

import { getGetV2ListChatConnectionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";

import type { SubscriptionOption } from "./helpers";

interface Props {
  option: SubscriptionOption;
  variant: "primary" | "secondary";
  onConnected: () => void;
}

/**
 * One sign-in button, its own component because `useOAuthConnect` is a hook
 * and the step offers however many subscriptions the deployment enables.
 */
export function ConnectOptionButton({ option, variant, onConnected }: Props) {
  const queryClient = useQueryClient();
  const { connect, isPending } = useOAuthConnect({
    provider: option.authProvider,
    onSuccess: () => {
      // The next step's copy depends on what is connected, and so does the
      // decision to show this step at all on a later visit.
      queryClient.invalidateQueries({
        queryKey: getGetV2ListChatConnectionsQueryKey(),
      });
      onConnected();
    },
  });

  return (
    <Button
      variant={variant}
      size="large"
      onClick={connect}
      loading={isPending}
      rightIcon={<Icon icon={LinkSquare01Icon} size={18} />}
    >
      {option.connectLabel}
    </Button>
  );
}
