"use client";

import { ArrowSquareOutIcon, SpinnerGapIcon, XIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { useDeviceAuthConnect } from "./useDeviceAuthConnect";

interface Props {
  provider: string;
  providerName: string;
  onSuccess: () => void;
}

export function DeviceAuthConnectButton({
  provider,
  providerName,
  onSuccess,
}: Props) {
  const { connect, cancel, phase, userCode, verificationUrl, isPending } =
    useDeviceAuthConnect({ provider, onSuccess });

  if (phase === "idle" || phase === "error" || phase === "done") {
    return (
      <div className="flex flex-col gap-3">
        <Text variant="body" className="text-[#505057]">
          {providerName} uses device authorization. Click below, then follow the
          link to approve access.
        </Text>
        <Button
          type="button"
          variant="primary"
          size="large"
          onClick={connect}
          loading={false}
        >
          Connect {providerName}
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-[#505057]">
        Open the link below and enter the code to connect your {providerName}{" "}
        account.
      </Text>

      <div className="flex flex-col gap-3 rounded-lg border border-[#E0E0E3] bg-[#F9F9FA] p-4">
        <div className="flex flex-col gap-1">
          <Text variant="small" className="font-medium text-[#83838C]">
            Your code
          </Text>
          <Text
            variant="h3"
            as="p"
            className="select-all text-center font-mono text-2xl tracking-widest text-[#1F1F20]"
          >
            {userCode}
          </Text>
        </div>

        <a
          href={verificationUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-center gap-2 rounded-md bg-[#1F1F20] px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#2F2F30]"
        >
          Open {providerName}
          <ArrowSquareOutIcon size={16} />
        </a>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-[#83838C]">
          <SpinnerGapIcon size={16} className="animate-spin" />
          <Text variant="small">Waiting for approval…</Text>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="small"
          onClick={cancel}
          rightIcon={<XIcon size={14} />}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}
