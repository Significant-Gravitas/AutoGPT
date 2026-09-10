"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

import { useOAuthConnect } from "./useOAuthConnect";
import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  provider: string;
  providerName: string;
  buttonLabel?: string;
  /** Whose terms a linked run falls under, where that is not AutoGPT's. The
   *  sign-in window is a product ("ChatGPT") and the terms are a company's
   *  ("OpenAI"), so this is named separately rather than reusing the label
   *  on the button. */
  termsNotice?: string;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

export function OAuthConnectButton({
  provider,
  providerName,
  buttonLabel,
  termsNotice,
  onSuccess,
}: Props) {
  const { connect, isPending } = useOAuthConnect({ provider, onSuccess });

  return (
    <div className="flex flex-col gap-3">
      <Text variant="body" className="text-[#505057]">
        We&apos;ll open a {providerName} sign-in window. Approve access there to
        finish connecting.
      </Text>
      <Button
        type="button"
        variant="primary"
        size="large"
        onClick={connect}
        loading={isPending}
        rightIcon={<Icon icon={LinkSquare01Icon} size={18} />}
      >
        {buttonLabel ?? `Continue with ${providerName}`}
      </Button>
      {termsNotice && (
        <Text variant="small" className="text-[#8A8A90]">
          Linked runs are sent to {termsNotice} under your own account and
          follow {termsNotice}&apos;s terms.
        </Text>
      )}
    </div>
  );
}
