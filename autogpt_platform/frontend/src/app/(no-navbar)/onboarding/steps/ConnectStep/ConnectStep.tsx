"use client";

import Link from "next/link";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Text } from "@/components/atoms/Text/Text";

import { ConnectOptionButton } from "./ConnectOptionButton";
import { useConnectStep } from "./useConnectStep";

/**
 * The first thing a self-host install asks for: a model to run on.
 *
 * A fresh install has no way to answer a single message until someone
 * configures a provider, and the honest shape of that ask is not "paste an
 * API key" — most people setting this up already pay for a subscription that
 * can do the work. So the zero-config path leads and API keys become the
 * advanced one.
 *
 * Which subscriptions appear is the deployment's answer, not this file's:
 * the copy names whatever the server offers rather than one provider it was
 * written around.
 *
 * Deliberately skippable. A user who wants API keys, or who has already
 * configured them, should not have to link an account to get past a wizard.
 */
export function ConnectStep() {
  const { skip, isAlreadyLinked, options } = useConnectStep();

  return (
    <FadeIn>
      <div className="flex w-full max-w-lg flex-col items-center gap-8 px-4">
        <div className="flex flex-col items-center gap-3 text-center">
          <AutoGPTLogo
            className="relative right-[3rem] h-24 w-[12rem]"
            hideText
          />
          <Text variant="h3">Power your agents in one sign-in</Text>
          <Text variant="lead" as="span" className="!text-zinc-500">
            {isAlreadyLinked
              ? "Your subscription is connected. Agents will run on it instead of spending AutoGPT credits."
              : `Connect ${namesOf(options)} and AutoGPT runs with no API keys and no billing setup.`}
            {!isAlreadyLinked && options.length === 1 && options[0].models
              ? ` You get ${options[0].models}.`
              : ""}
          </Text>
        </div>

        <div className="flex w-full flex-col items-center gap-4">
          {isAlreadyLinked || options.length === 0 ? (
            <Button variant="primary" size="large" onClick={skip}>
              Continue
            </Button>
          ) : (
            <>
              {options.map((option, index) => (
                <ConnectOptionButton
                  key={option.authProvider}
                  option={option}
                  variant={index === 0 ? "primary" : "secondary"}
                  onConnected={skip}
                />
              ))}
              <Text
                variant="small"
                as="span"
                className="uppercase tracking-[0.08em] !text-zinc-400"
              >
                or configure manually
              </Text>
              <Button variant="secondary" size="large" onClick={skip}>
                Skip for now
              </Button>
              <Text variant="small" as="span" className="!text-zinc-400">
                You can add API keys in Settings &rarr; Integrations at any
                time.
              </Text>
            </>
          )}
        </div>

        {/* Named here rather than discovered later: someone arriving with a
            Claude subscription will look for it, and the answer is a policy
            they cannot change, not a missing feature. */}
        <Text variant="small" as="span" className="text-center !text-zinc-400">
          Claude subscriptions can&apos;t be linked — Anthropic blocks
          third-party subscription access. Anthropic models work via{" "}
          <Link
            href="/settings/integrations"
            className="underline underline-offset-2"
          >
            API key
          </Link>
          .
        </Text>
      </div>
    </FadeIn>
  );
}

/** "the ChatGPT plan you already have", or "a ChatGPT or Grok plan you
 *  already have" — the promise names what is actually on offer. */
function namesOf(options: { displayName: string }[]): string {
  const names = options.map((option) => option.displayName);
  if (names.length === 0) return "a subscription you already have";
  if (names.length === 1) return `the ${names[0]} plan you already have`;
  const list = `${names.slice(0, -1).join(", ")} or ${names[names.length - 1]}`;
  return `a ${list} plan you already have`;
}
