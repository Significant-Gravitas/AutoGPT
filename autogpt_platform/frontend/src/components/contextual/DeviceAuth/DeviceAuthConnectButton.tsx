"use client";

import {
  Cancel01Icon,
  LinkSquare01Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

import { useDeviceAuthConnect } from "./useDeviceAuthConnect";

interface Props {
  provider: string;
  providerName: string;
  onSuccess: (credentials?: CredentialsMetaResponse) => void;
}

export function DeviceAuthConnectButton({
  provider,
  providerName,
  onSuccess,
}: Props) {
  const { connect, cancel, phase, userCode, verificationUrl } =
    useDeviceAuthConnect({ provider, onSuccess });

  // `awaiting_user` spans the initiate round-trip, when there is no code and
  // no URL yet. Rendering the code panel then shows an empty box above a link
  // whose href="" resolves to the current page.
  if (phase === "awaiting_user" && !userCode) {
    return (
      <div className="flex flex-col gap-3">
        <Text variant="body" className="text-zinc-600">
          Requesting a code from {providerName}…
        </Text>
        <div className="flex items-center gap-2 text-zinc-500">
          <Icon icon={Loading03Icon} size={16} className="animate-spin" />
          <Text variant="small">Starting device authorization…</Text>
        </div>
      </div>
    );
  }

  if (phase === "idle" || phase === "error" || phase === "done") {
    return (
      <div className="flex flex-col gap-3">
        <Text variant="body" className="text-zinc-600">
          {providerName} uses device authorization. Click below, then follow the
          link to approve access.
        </Text>
        <Button type="button" variant="primary" size="large" onClick={connect}>
          Connect {providerName}
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-zinc-600">
        Open the link below and enter the code to connect your {providerName}{" "}
        account.
      </Text>

      <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-zinc-50 p-4">
        <div className="flex flex-col gap-1">
          <Text variant="small" className="font-medium text-zinc-500">
            Your code
          </Text>
          {/* unmask={false}: Text unmasks for session replay by default, and
              this is a live authorization code — it must not be recorded. */}
          <Text
            variant="h3"
            as="p"
            unmask={false}
            className="select-all text-center font-mono text-2xl tracking-widest text-zinc-800"
          >
            {userCode}
          </Text>
        </div>

        <a
          href={verificationUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-center gap-2 rounded-md bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-zinc-800"
        >
          Open {providerName}
          <Icon icon={LinkSquare01Icon} size={16} />
        </a>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-zinc-500">
          <Icon icon={Loading03Icon} size={16} className="animate-spin" />
          <Text variant="small">Waiting for approval…</Text>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="small"
          onClick={cancel}
          rightIcon={<Icon icon={Cancel01Icon} size={14} />}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}
