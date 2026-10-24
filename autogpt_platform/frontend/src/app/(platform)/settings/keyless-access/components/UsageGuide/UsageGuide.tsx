"use client";

import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { Text } from "@/components/atoms/Text/Text";
import { environment } from "@/services/environment";

import type { Platform } from "../../helpers";
import { CodeSnippet } from "../CodeSnippet/CodeSnippet";
import { SiteSetup } from "./components/SiteSetup";
import { UsageStep } from "./components/UsageStep";
import {
  apiKeyCallSnippet,
  callSnippet,
  exchangeSnippet,
  fetchSnippet,
  tokenStep,
} from "./helpers";

interface Props {
  identity: WorkloadIdentityInfo;
  platform: Platform;
  showSiteSetup?: boolean;
}

export function UsageGuide({ identity, platform, showSiteSetup }: Props) {
  const baseUrl = environment.getAGPTServerBaseUrl();
  const step = tokenStep({ identity, platform });
  const minutes = Math.round((identity.token_ttl_seconds ?? 900) / 60);
  const isJavaScriptPlatform = ["vercel", "website", "custom"].includes(
    platform,
  );

  return (
    <div className="flex flex-col gap-5">
      <Text variant="body" className="text-zinc-700">
        Where you would paste an API key, your code instead proves where it runs
        and swaps that proof for a token that lasts {minutes} minutes. There is
        no secret to store, leak or rotate.
      </Text>

      {(platform === "website" || platform === "custom") && (
        <SiteSetup identity={identity} defaultOpen={showSiteSetup} />
      )}

      <UsageStep number={1} title="Get an identity token" intro={step.intro}>
        {step.snippets.map((snippet) => (
          <CodeSnippet
            key={snippet.label}
            label={snippet.label}
            code={snippet.code}
          />
        ))}
      </UsageStep>

      <UsageStep
        number={2}
        title="Swap it for an AutoGPT token"
        intro="The workload ID below is not a secret: it only names what AutoGPT should check the token against."
      >
        <CodeSnippet
          label="Terminal"
          code={exchangeSnippet(identity.id, baseUrl)}
        />
      </UsageStep>

      <UsageStep
        number={3}
        title="Call the API like you would with a key"
        intro="Send the token as a Bearer token. When it expires, run step 2 again."
      >
        <CodeSnippet label="Terminal" code={callSnippet(identity, baseUrl)} />
        <CodeSnippet
          label="The same call with an API key, for comparison"
          code={apiKeyCallSnippet(identity, baseUrl)}
        />
        {isJavaScriptPlatform && (
          <CodeSnippet
            label="Steps 2 and 3 in JavaScript"
            code={fetchSnippet(identity, baseUrl)}
          />
        )}
      </UsageStep>
    </div>
  );
}
