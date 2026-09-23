import { Button } from "@/components/atoms/Button/Button";
import {
  BubbleChatIcon,
  Cancel01Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  onClose: () => void;
  canRevert: boolean;
  revertTargetVersion: number | null;
  onRevert: () => void;
}

export function PanelHeader({
  onClose,
  canRevert,
  revertTargetVersion,
  onRevert,
}: Props) {
  return (
    <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
      <div className="flex items-center gap-2">
        <Icon icon={BubbleChatIcon} size={18} className="text-violet-600" />
        <span className="text-sm font-semibold text-slate-800">
          Chat with Builder
        </span>
      </div>
      <div className="flex items-center gap-1">
        {canRevert && (
          <Button
            variant="ghost"
            size="small"
            onClick={onRevert}
            leftIcon={<Icon icon={RefreshIcon} size={14} />}
            aria-label={
              revertTargetVersion != null
                ? `Revert to version ${revertTargetVersion}`
                : "Revert to previous version"
            }
            title="Revert to the graph version that was active before the last edit"
          >
            Revert
          </Button>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Close"
        >
          <Icon icon={Cancel01Icon} size={16} />
        </Button>
      </div>
    </div>
  );
}
