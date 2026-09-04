"use client";

import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

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
 * Deliberately skippable. A user who wants API keys, or who has already
 * configured them, should not have to link an account to get past a wizard.
 */
export function ConnectStep() {
  const { connect, isConnecting, skip, isAlreadyLinked, models } =
    useConnectStep();

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
              ? "Your ChatGPT plan is connected. Agents will run on it instead of spending AutoGPT credits."
              : "Connect the ChatGPT plan you already have and AutoGPT runs with no API keys and no billing setup."}
            {models && !isAlreadyLinked ? ` You get ${models}.` : ""}
          </Text>
        </div>

        <div className="flex w-full flex-col items-center gap-4">
          {isAlreadyLinked ? (
            <Button variant="primary" size="large" onClick={skip}>
              Continue
            </Button>
          ) : (
            <Button
              variant="primary"
              size="large"
              onClick={connect}
              loading={isConnecting}
              rightIcon={<Icon icon={LinkSquare01Icon} size={18} />}
            >
              Sign in with ChatGPT
            </Button>
          )}

          {!isAlreadyLinked && (
            <>
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
