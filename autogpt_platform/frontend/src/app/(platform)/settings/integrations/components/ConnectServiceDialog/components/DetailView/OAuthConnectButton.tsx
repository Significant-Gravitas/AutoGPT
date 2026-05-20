"use client";

import { ArrowSquareOutIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { useOAuthConnect } from "./useOAuthConnect";

interface Props {
  provider: string;
  providerName: string;
  onSuccess: () => void;
}

export function OAuthConnectButton({
  provider,
  providerName,
  onSuccess,
}: Props) {
  const { connect, isPending } = useOAuthConnect({ provider, onSuccess });

  return (
    <div className="flex flex-col gap-3">
      <Text variant="body" className="text-[#505057]">
        We&apos;ll open {providerName} in a popup. After approving, you&apos;ll
        come right back.
      </Text>
      <Button
        type="button"
        variant="primary"
        size="large"
        onClick={connect}
        loading={isPending}
        rightIcon={<ArrowSquareOutIcon size={18} />}
      >
        Continue with {providerName}
      </Button>
    </div>
  );
}
