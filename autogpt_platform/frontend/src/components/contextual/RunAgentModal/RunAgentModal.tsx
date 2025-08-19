import { Badge } from "@/components/atoms/Badge/Badge";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  triggerSlot: React.ReactNode;
}

export function RunAgentModal({ triggerSlot }: Props) {
  return (
    <Dialog>
      <Dialog.Trigger>{triggerSlot}</Dialog.Trigger>
      <Dialog.Content>
        <Badge variant="info">New run</Badge>
      </Dialog.Content>
    </Dialog>
  );
}
