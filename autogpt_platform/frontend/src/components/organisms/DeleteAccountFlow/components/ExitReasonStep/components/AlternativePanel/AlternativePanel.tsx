import { Button } from "@/components/atoms/Button/Button";
import { createIconComponent } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Alert } from "@/components/molecules/Alert/Alert";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import type { PanelActionKind, ReasonPanel } from "../../../../helpers";
import { useAlternativePanel } from "./useAlternativePanel";

// Alert's `icon` prop takes a component (its defaults come from lucide).
const InfoIcon = createIconComponent(InformationCircleIcon);

interface Props {
  panel: ReasonPanel;
  isActionPending: boolean;
  onAction: (kind: PanelActionKind) => void;
}

export function AlternativePanel({ panel, isActionPending, onAction }: Props) {
  const { action } = panel;
  const { panelRef } = useAlternativePanel(Boolean(action));

  return (
    <Alert
      ref={panelRef}
      role="note"
      tabIndex={-1}
      icon={InfoIcon}
      className="focus:outline-none [&>svg]:top-3.5 [&>svg]:translate-y-0"
    >
      <div className="flex flex-col items-start gap-3">
        <Text variant="body" tone="emphasis">
          {panel.copy}
        </Text>
        {action ? (
          <Button
            variant="secondary"
            size="small"
            loading={isActionPending}
            onClick={() => onAction(action.kind)}
            className="h-auto min-h-11 whitespace-normal text-left lg:min-h-[2.25rem]"
          >
            {action.label}
          </Button>
        ) : null}
      </div>
    </Alert>
  );
}
