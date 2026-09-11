"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowClockwiseIcon,
  CheckCircleIcon,
  DesktopTowerIcon,
} from "@phosphor-icons/react";

interface Props {
  isRefreshing: boolean;
  onRefresh: () => void;
}

const REQUIRED_SETUP_STEPS = [
  {
    title: "Install the Local PC executor",
    command:
      "pipx install git+https://github.com/Significant-Gravitas/autogpt-local-executor.git",
  },
  {
    title: "Sign in to AutoGPT",
    command: "autogpt-shim auth",
  },
] as const;

const CONNECTION_OPTIONS = [
  {
    title: "Start it now",
    command: "autogpt-shim start",
    detail: "Leave this command running while you use Local PC.",
  },
  {
    title: "Start automatically at sign-in",
    command: "autogpt-shim install",
    detail: "Then run the OS-specific enable command it prints.",
  },
] as const;

export function LocalExecutorSetup({ isRefreshing, onRefresh }: Props) {
  return (
    <section
      aria-labelledby="local-executor-setup-title"
      className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4"
    >
      <div className="flex items-start gap-3">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-full bg-violet-100 text-violet-700">
          <DesktopTowerIcon size={20} weight="fill" aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-1">
          <Text
            id="local-executor-setup-title"
            variant="body-medium"
            as="h3"
            className="text-zinc-900"
          >
            Connect a Computer
          </Text>
          <Text variant="small" className="mt-1 text-zinc-600">
            Run these commands on the Mac, Windows PC, or Linux computer you
            want AutoGPT to use. It connects securely to AutoGPT, so you can
            choose its folders here even when this chat is open on another
            device.
          </Text>
        </div>
      </div>

      <ol className="mt-4 flex flex-col gap-3">
        {REQUIRED_SETUP_STEPS.map((step, index) => (
          <li key={step.command} className="flex items-start gap-3">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-white text-xs font-semibold text-zinc-700 ring-1 ring-zinc-200">
              {index + 1}
            </span>
            <div className="min-w-0 flex-1">
              <Text variant="small-medium" className="text-zinc-800">
                {step.title}
              </Text>
              <pre className="mt-1 max-w-full overflow-x-auto rounded-lg bg-zinc-900 px-3 py-2 text-left font-mono text-xs text-zinc-50">
                <code translate="no">{step.command}</code>
              </pre>
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-4 border-t border-zinc-200 pt-4">
        <Text variant="small-medium" className="text-zinc-800">
          Keep the computer connected
        </Text>
        <Text variant="small" className="mt-1 text-zinc-600">
          Choose one of these options:
        </Text>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {CONNECTION_OPTIONS.map((option) => (
            <div
              key={option.command}
              className="min-w-0 rounded-xl border border-zinc-200 bg-white p-3"
            >
              <Text variant="small-medium" className="text-zinc-800">
                {option.title}
              </Text>
              <pre className="mt-2 max-w-full overflow-x-auto rounded-lg bg-zinc-900 px-3 py-2 text-left font-mono text-xs text-zinc-50">
                <code translate="no">{option.command}</code>
              </pre>
              <Text variant="small" className="mt-2 text-zinc-600">
                {option.detail}
              </Text>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-zinc-200 pt-4">
        <div className="flex items-center gap-2 text-xs text-zinc-600">
          <CheckCircleIcon
            size={16}
            weight="fill"
            className="text-green-600"
            aria-hidden="true"
          />
          Waiting for a signed-in executor…
        </div>
        <Button
          type="button"
          variant="secondary"
          size="small"
          loading={isRefreshing}
          leftIcon={
            <ArrowClockwiseIcon size={16} weight="bold" aria-hidden="true" />
          }
          onClick={onRefresh}
        >
          Check Again
        </Button>
      </div>
    </section>
  );
}
