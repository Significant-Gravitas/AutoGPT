import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { ToolInvocation } from "./useWorkDoneCounters";

interface Props {
  title: string;
  invocations: ToolInvocation[];
  onClose: () => void;
}

export function ToolInvocationsDialog({ title, invocations, onClose }: Props) {
  return (
    <Dialog
      title={title}
      controlled={{
        isOpen: true,
        set: function handleOpenChange(open) {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-2 p-4">
          {invocations.length === 0 && (
            <p className="text-sm text-neutral-500">No invocations recorded.</p>
          )}
          {invocations.map(function renderInvocation(inv, i) {
            return (
              <div
                key={`${inv.toolName}-${i}`}
                className="flex flex-col gap-0.5 rounded-md bg-neutral-50 px-3 py-2"
              >
                <span className="text-sm font-medium text-neutral-700">
                  {inv.toolName}
                </span>
                {inv.argsSummary && (
                  <span className="text-xs text-neutral-500">
                    {inv.argsSummary}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
