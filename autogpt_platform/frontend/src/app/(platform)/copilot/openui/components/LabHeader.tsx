import { Icon } from "@/components/atoms/Icon/Icon";
import { TestTube01Icon, ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";

export function LabHeader() {
  return (
    <>
      <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-zinc-200 px-5 sm:px-8">
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <Icon icon={TestTube01Icon} size={16} />
          <span className="hidden sm:inline">Labs</span>
          <span className="hidden px-1 text-zinc-300 sm:inline">/</span>
          <span className="font-medium text-zinc-800">
            Generative workspace
          </span>
        </div>
        <Button
          as="NextLink"
          href="https://www.openui.com"
          target="_blank"
          rel="noopener noreferrer"
          variant="ghost"
          size="xs"
          rightIcon={<Icon icon={ArrowUpRight01Icon} size={13} />}
        >
          About OpenUI
        </Button>
      </header>
      <div className="flex flex-wrap items-end justify-between gap-4 px-5 py-5 sm:px-8">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <span className="size-1.5 rounded-full bg-purple-500" />
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-purple-600">
              The interface follows your intent
            </p>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-900">
            From answers to action.
          </h1>
          <p className="mt-2 text-sm text-zinc-500">
            An agent response you can explore, edit, and build on.
          </p>
        </div>
        <span className="hidden rounded-full border border-zinc-200 px-3 py-1.5 text-[11px] text-zinc-500 md:inline">
          AutoGPT × OpenUI <span className="mx-1 text-zinc-300">/</span>{" "}
          Experiment
        </span>
      </div>
    </>
  );
}
