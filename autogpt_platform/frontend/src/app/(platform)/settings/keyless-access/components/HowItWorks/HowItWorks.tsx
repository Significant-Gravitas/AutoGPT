"use client";

import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { environment } from "@/services/environment";

import { KEYLESS_DOCS_URL } from "../../helpers";
import { CodeSnippet } from "../CodeSnippet/CodeSnippet";

const STEPS = [
  {
    title: "Your platform vouches for your code",
    body: "GitHub, Vercel, Google Cloud or your own site signs a short-lived token that names the exact repository, project or server that is running.",
  },
  {
    title: "AutoGPT checks it",
    body: "We verify the signature with the platform's public key, check that the token names a workload you trust here, and hand back an AutoGPT token.",
  },
  {
    title: "Use it like an API key",
    body: "Send it in the Authorization header. It expires within the hour, so there is nothing to store, leak or rotate.",
  },
];

export function HowItWorks() {
  const url = `${environment.getAGPTServerBaseUrl()}/external-api/v1/me`;

  return (
    <section className="flex flex-col gap-4 rounded-2xl border border-zinc-200 bg-white p-5">
      <Text variant="large-medium" as="h2" className="text-textBlack">
        How it works
      </Text>
      <ol className="grid gap-4 sm:grid-cols-3">
        {STEPS.map((step, index) => (
          <li key={step.title} className="flex flex-col gap-1">
            <Text variant="body-medium" as="span" className="text-textBlack">
              {index + 1}. {step.title}
            </Text>
            <Text variant="small" as="span" className="text-zinc-600">
              {step.body}
            </Text>
          </li>
        ))}
      </ol>
      <div className="grid gap-2 md:grid-cols-2">
        <CodeSnippet
          label="With an API key"
          code={`curl ${url} \\\n  -H "X-API-Key: $AUTOGPT_API_KEY"`}
        />
        <CodeSnippet
          label="Keyless"
          code={`curl ${url} \\\n  -H "Authorization: Bearer $AUTOGPT_TOKEN"`}
        />
      </div>
      <Link href={KEYLESS_DOCS_URL} isExternal variant="secondary">
        Read the keyless access guide
      </Link>
    </section>
  );
}
