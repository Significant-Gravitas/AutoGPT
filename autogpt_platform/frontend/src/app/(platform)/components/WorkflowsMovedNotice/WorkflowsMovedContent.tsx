import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import * as Dialog from "@radix-ui/react-dialog";
import { RefObject } from "react";
import { WorkflowsMovedWalkthrough } from "./WorkflowsMovedWalkthrough";

interface Props {
  titleRef: RefObject<HTMLHeadingElement>;
  onDismiss: () => void;
}

export function WorkflowsMovedContent({ titleRef, onDismiss }: Props) {
  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="px-6 pb-3 pt-5 sm:px-8">
          <p className="mb-1.5 text-xs font-medium uppercase tracking-widest text-violet-700">
            Workspace update
          </p>
          <Dialog.Title
            ref={titleRef}
            tabIndex={-1}
            className="pr-7 text-2xl font-semibold leading-tight tracking-tight text-zinc-900 outline-none sm:text-[1.625rem]"
          >
            Your agents have moved
          </Dialog.Title>
          <Dialog.Description className="mt-1.5 text-sm leading-relaxed text-zinc-600">
            Agents are now{" "}
            <strong className="font-medium text-zinc-900">workflows</strong>.
            Find yours under{" "}
            <strong className="font-medium text-zinc-900">Otto</strong> in Team.
          </Dialog.Description>
          <WorkflowLocation />
        </div>
        <div className="px-4 pb-3 sm:px-6">
          <WorkflowsMovedWalkthrough />
        </div>
      </div>
      <WorkflowsMovedActions onDismiss={onDismiss} />
    </>
  );
}

function WorkflowsMovedActions({ onDismiss }: Pick<Props, "onDismiss">) {
  return (
    <div className="shrink-0 border-t border-zinc-100 bg-white px-6 py-2 sm:px-8">
      <p className="mb-1.5 text-xs leading-none text-zinc-500">
        Nothing to set up again.
      </p>
      <div className="flex flex-col gap-2 sm:flex-row-reverse">
        <Button
          as="NextLink"
          href="/team/autopilot?tab=workflows"
          onClick={onDismiss}
          rightIcon={<Icon icon={ArrowRight02Icon} size={18} aria-hidden />}
        >
          Go to my workflows
        </Button>
        <Button variant="ghost" onClick={onDismiss}>
          Got it
        </Button>
      </div>
    </div>
  );
}

function WorkflowLocation() {
  return (
    <ol
      aria-label="Where to find your workflows"
      className="mt-3 flex w-fit flex-wrap items-center gap-2 rounded-full border border-zinc-200/80 bg-zinc-50 px-3 py-1 text-xs text-zinc-600"
    >
      {["Team", "Otto", "Workflows"].map((label, index) => (
        <li key={label} className="flex items-center gap-2">
          {index > 0 && (
            <Icon
              icon={ArrowRight02Icon}
              size={12}
              className="text-zinc-400"
              aria-hidden
            />
          )}
          <span
            className={index === 2 ? "font-medium text-violet-700" : undefined}
          >
            {label}
          </span>
        </li>
      ))}
    </ol>
  );
}
