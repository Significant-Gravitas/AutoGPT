import { Button } from "@/components/atoms/Button/Button";
import { ArrowCounterClockwise, ChatCircle, X } from "@phosphor-icons/react";

interface Props {
  onClose: () => void;
  undoCount: number;
  onUndo: () => void;
}

export function PanelHeader({ onClose, undoCount, onUndo }: Props) {
  return (
    <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
      <div className="flex items-center gap-2">
        <ChatCircle size={18} weight="fill" className="text-violet-600" />
        <span className="text-sm font-semibold text-slate-800">
          Chat with Builder
        </span>
      </div>
      <div className="flex items-center gap-1">
        {undoCount > 0 && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onUndo}
            aria-label="Undo last applied change"
            title="Undo last applied change"
          >
            <ArrowCounterClockwise size={16} />
          </Button>
        )}
        <Button variant="icon" size="icon" onClick={onClose} aria-label="Close">
          <X size={16} />
        </Button>
      </div>
    </div>
  );
}
