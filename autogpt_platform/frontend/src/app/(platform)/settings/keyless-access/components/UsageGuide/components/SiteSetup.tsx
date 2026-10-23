"use client";

import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { Text } from "@/components/atoms/Text/Text";
import { Collapsible } from "@/components/molecules/Collapsible/Collapsible";

import { CodeSnippet } from "../../CodeSnippet/CodeSnippet";
import { siteSetupSnippets } from "../helpers";

interface Props {
  identity: WorkloadIdentityInfo;
  defaultOpen?: boolean;
}

export function SiteSetup({ identity, defaultOpen }: Props) {
  const snippets = siteSetupSnippets(identity);

  return (
    <Collapsible
      defaultOpen={defaultOpen}
      className="rounded-lg border border-zinc-200 p-3"
      trigger={
        <Text variant="body-medium" as="span" className="text-textBlack">
          One-time setup when {identity.issuer} is your own site
        </Text>
      }
    >
      <div className="flex flex-col gap-3 pt-3">
        <Text variant="small" className="text-zinc-600">
          Your site becomes the issuer: it publishes a public key, and your
          server signs tokens with the matching private key. The private key
          never leaves your server, and AutoGPT only ever reads the public one.
        </Text>
        <CodeSnippet
          label="1. Make a key pair (Node.js, npm install jose)"
          code={snippets.keygen}
        />
        <CodeSnippet
          label={`2. Publish at ${snippets.discoveryUrl}`}
          code={snippets.discovery}
        />
        <Text variant="small" className="text-zinc-600">
          {`3. Publish the printed key set at ${identity.issuer}/.well-known/jwks.json. To rotate the key later, publish the new one there; nothing changes on AutoGPT.`}
        </Text>
      </div>
    </Collapsible>
  );
}
