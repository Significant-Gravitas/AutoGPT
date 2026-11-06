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
        "pipx install git+https://github.com/Significant-Gravitas/autogpt-local-executor.git@d9265c724b6e9e242e42ca2e276be4d93d150686",
    },
    {
      title: "Sign in to this AutoGPT deployment",
      command: deployment?.authCommand,
    },
  ];
  return (
    <section
      aria-labelledby="local-executor-setup-title"
      className="rounded-2xl border border-border-base bg-surface-muted p-4"
    >
      <div className="flex items-start gap-3">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-full bg-accent-surface text-accent-strong">
          <Icon icon={ComputerIcon} size={20} aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-1">
          <Text
            id="local-executor-setup-title"
            variant="body-medium"
            as="h3"
            tone="primary"
          >
            Connect a Computer
          </Text>
          <Text variant="small" tone="secondary" className="mt-1">
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
            <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-surface text-caption font-semibold text-foreground-body ring-1 ring-border-base">
              {index + 1}
            </span>
            <div className="min-w-0 flex-1">
              <Text variant="small-medium" tone="emphasis">
                {step.title}
              </Text>
              <pre className="mt-1 max-w-full overflow-x-auto rounded-lg bg-action-hover px-3 py-2 text-left font-mono text-caption text-action-foreground">
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
        className="mt-3 inline-block text-body text-accent-strong underline underline-offset-2"
      >
        Install pipx for your operating system
      </a>

      <div className="mt-4 border-t border-border-base pt-4">
        <Text variant="small-medium" tone="emphasis">
          Keep the computer connected
        </Text>
        <Text variant="small" tone="secondary" className="mt-1">
          Run this in your terminal (PowerShell on Windows):
        </Text>
        <pre className="mt-2 max-w-full overflow-x-auto rounded-lg bg-action-hover px-3 py-2 text-left font-mono text-caption text-action-foreground">
          <code translate="no">
            {deployment?.startCommand ?? "Loading deployment settings…"}
          </code>
        </pre>
        <Text variant="small" tone="secondary" className="mt-2">
          Leave it running while you use Local PC. Shell commands and computer
          control remain off unless you explicitly enable them on that computer.
          Both addresses must be reachable from the computer you connect;
          localhost works only when AutoGPT runs on that same computer. Use
          HTTPS for a deployment on another computer.
        </Text>
        <details className="mt-3 text-body text-foreground-body">
          <summary className="cursor-pointer font-medium">
            Start automatically at sign-in
          </summary>
          <Text variant="small" tone="secondary" className="mt-2">
            Save the settings below in your executor&apos;s config.toml, then
            install the per-user service. Keep the file at this location:
          </Text>
          <ul className="my-2 list-inside list-disc text-caption break-all">
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
          <pre className="max-w-full overflow-x-auto rounded-lg bg-action-hover px-3 py-2 text-caption text-action-foreground">
            <code translate="no">
              {deployment?.config ?? "Loading deployment settings…"}
            </code>
          </pre>
          <pre className="mt-2 max-w-full overflow-x-auto rounded-lg bg-action-hover px-3 py-2 text-caption text-action-foreground">
            <code translate="no">autogpt-shim install</code>
          </pre>
          <Text variant="small" tone="secondary" className="mt-2">
            Run the OS-specific enable command it prints.
          </Text>
        </details>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-border-base pt-4">
        <div className="flex items-center gap-2 text-caption text-foreground-secondary">
          <span className="flex text-success">
            <Icon icon={CheckmarkCircle02Icon} size={16} aria-hidden="true" />
          </span>
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
