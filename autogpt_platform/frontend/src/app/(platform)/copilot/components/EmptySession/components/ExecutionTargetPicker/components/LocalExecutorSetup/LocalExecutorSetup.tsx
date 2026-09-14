"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  RefreshIcon,
  CheckmarkCircle02Icon,
  ComputerIcon,
} from "@hugeicons/core-free-icons";
import { useLocalExecutorSetup } from "./useLocalExecutorSetup";

interface Props {
  isRefreshing: boolean;
  onRefresh: () => void;
}

export function LocalExecutorSetup({ isRefreshing, onRefresh }: Props) {
  const { deployment } = useLocalExecutorSetup();
  const setupSteps = [
    {
      title:
        "Install the Local PC executor (Python 3.11+, pipx, and Git required)",
      command:
        "pipx install git+https://github.com/Significant-Gravitas/autogpt-local-executor.git@038d424caa3ed74c183194be53b7bd64eee58c44",
    },
    {
      title: "Sign in to this AutoGPT deployment",
      command: deployment?.authCommand,
    },
  ];
  return (
    <section
      aria-labelledby="local-executor-setup-title"
      className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4"
    >
      <div className="flex items-start gap-3">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-full bg-violet-100 text-violet-700">
          <Icon icon={ComputerIcon} size={20} aria-hidden="true" />
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
        {setupSteps.map((step, index) => (
          <li key={step.title} className="flex items-start gap-3">
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-white text-xs font-semibold text-zinc-700 ring-1 ring-zinc-200">
              {index + 1}
            </span>
            <div className="min-w-0 flex-1">
              <Text variant="small-medium" className="text-zinc-800">
                {step.title}
              </Text>
              <pre className="mt-1 max-w-full overflow-x-auto rounded-lg bg-zinc-900 px-3 py-2 text-left font-mono text-xs text-zinc-50">
                <code translate="no">
                  {step.command ?? "Loading deployment settings…"}
                </code>
              </pre>
            </div>
          </li>
        ))}
      </ol>

      <a
        href="https://pipx.pypa.io/latest/how-to/install-pipx.html"
        target="_blank"
        rel="noopener noreferrer"
        className="mt-3 inline-block text-sm text-violet-700 underline underline-offset-2"
      >
        Install pipx for your operating system
      </a>

      <div className="mt-4 border-t border-zinc-200 pt-4">
        <Text variant="small-medium" className="text-zinc-800">
          Keep the computer connected
        </Text>
        <Text variant="small" className="mt-1 text-zinc-600">
          Run this in your terminal (PowerShell on Windows):
        </Text>
        <pre className="mt-2 max-w-full overflow-x-auto rounded-lg bg-zinc-900 px-3 py-2 text-left font-mono text-xs text-zinc-50">
          <code translate="no">
            {deployment?.startCommand ?? "Loading deployment settings…"}
          </code>
        </pre>
        <Text variant="small" className="mt-2 text-zinc-600">
          Leave it running while you use Local PC. Shell commands and computer
          control remain off unless you explicitly enable them on that computer.
          Both addresses must be reachable from the computer you connect;
          localhost works only when AutoGPT runs on that same computer. Use
          HTTPS for a deployment on another computer.
        </Text>
        <details className="mt-3 text-sm text-zinc-700">
          <summary className="cursor-pointer font-medium">
            Start automatically at sign-in
          </summary>
          <Text variant="small" className="mt-2 text-zinc-600">
            Save the settings below in your executor&apos;s config.toml, then
            install the per-user service. Keep the file at this location:
          </Text>
          <ul className="my-2 list-inside list-disc break-all text-xs">
            <li>
              macOS: ~/Library/Application
              Support/autogpt-local-executor/config.toml
            </li>
            <li>
              Linux: ~/.config/autogpt-local-executor/config.toml (replace
              ~/.config with $XDG_CONFIG_HOME if set)
            </li>
            <li>Windows: %APPDATA%\autogpt-local-executor\config.toml</li>
          </ul>
          <pre className="max-w-full overflow-x-auto rounded-lg bg-zinc-900 px-3 py-2 text-xs text-zinc-50">
            <code translate="no">
              {deployment?.config ?? "Loading deployment settings…"}
            </code>
          </pre>
          <pre className="mt-2 max-w-full overflow-x-auto rounded-lg bg-zinc-900 px-3 py-2 text-xs text-zinc-50">
            <code translate="no">autogpt-shim install</code>
          </pre>
          <Text variant="small" className="mt-2 text-zinc-600">
            Run the OS-specific enable command it prints.
          </Text>
        </details>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-zinc-200 pt-4">
        <div className="flex items-center gap-2 text-xs text-zinc-600">
          <Icon
            icon={CheckmarkCircle02Icon}
            size={16}
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
          leftIcon={<Icon icon={RefreshIcon} size={16} aria-hidden="true" />}
          onClick={onRefresh}
        >
          Check Again
        </Button>
      </div>
    </section>
  );
}
