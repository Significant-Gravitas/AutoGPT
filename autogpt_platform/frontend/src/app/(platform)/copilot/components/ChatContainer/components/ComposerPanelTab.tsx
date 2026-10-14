"use client";

import type { ContextPanelExpert } from "../../../store";
import { IntegrationsToggle } from "../../ContextPanel/components/IntegrationsToggle/IntegrationsToggle";
import { WorkspaceFilesTrigger } from "../../WorkspaceFilesTrigger/WorkspaceFilesTrigger";

interface Props {
  expert: ContextPanelExpert | null;
}

const buttonClassName = "size-9 shrink-0 rounded-full hover:bg-zinc-200/70";

/** The small tab on the composer's top-right edge that holds the files and
 *  integrations buttons on small viewports. It shares the task list's
 *  surface (TaskProgressBar), so the two read as parts of the composer. */
export function ComposerPanelTab({ expert }: Props) {
  return (
    <div className="flex justify-end pr-6 lg:hidden">
      <div className="flex items-center gap-0.5 rounded-t-2xl border border-b-0 border-zinc-200 bg-neutral-100 px-1.5 pt-1 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.9),inset_0_5px_6px_-4px_rgba(255,255,255,0.7)]">
        <WorkspaceFilesTrigger className={buttonClassName} />
        {expert && (
          <IntegrationsToggle
            expert={expert}
            size="lg"
            className={buttonClassName}
          />
        )}
      </div>
    </div>
  );
}
