import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import * as Dialog from "@radix-ui/react-dialog";
import { RefObject } from "react";
import { WorkflowsMovedWalkthrough } from "./WorkflowsMovedWalkthrough";

interface Props {
  titleRef: RefObject<HTMLHeadingElement | null>;
  onDismiss: () => void;
}

export function WorkflowsMovedContent({ titleRef, onDismiss }: Props) {
  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="px-6 pb-3 pt-5 sm:px-8">
          <p className="mb-1.5 text-xs font-medium uppercase tracking-widest text-accent-strong">
            Workspace update
          </p>
          <Dialog.Title
            ref={titleRef}
            tabIndex={-1}
            // TODO(tokens): SECRT-3152, rare size sm:text-[1.625rem]
            className="pr-7 text-2xl font-semibold leading-tight tracking-tight text-foreground-strong outline-none sm:text-[1.625rem]"
          >
            Your agents have moved
          </Dialog.Title>
          <Dialog.Description className="mt-1.5 text-sm leading-relaxed text-foreground-secondary">
            Agents are now{" "}
            <strong className="font-medium text-foreground-strong">
              workflows
            </strong>
            . Find yours under{" "}
            <strong className="font-medium text-foreground-strong">Otto</strong>{" "}
            in Team.
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
    <div className="shrink-0 border-t border-separator bg-surface px-6 py-2 sm:px-8">
      <p className="mb-1.5 text-xs leading-none text-foreground-tertiary">
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
      className="mt-3 flex w-fit flex-wrap items-center gap-2 rounded-full border border-border-base/80 bg-surface-muted px-3 py-1 text-xs text-foreground-secondary"
    >
      {["Team", "Otto", "Workflows"].map((label, index) => (
        <li key={label} className="flex items-center gap-2">
          {index > 0 && (
            <Icon
              icon={ArrowRight02Icon}
              size={12}
              className="text-foreground-subtle"
              aria-hidden
            />
          )}
          <span
            className={
              index === 2 ? "font-medium text-accent-strong" : undefined
            }
          >
            {label}
          </span>
        </li>
      ))}
    </ol>
  );
}
