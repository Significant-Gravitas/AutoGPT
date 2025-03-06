import { Button } from "@/components/agptui/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export default function DeleteConfirmDialog({
  entityType,
  entityName,
  open,
  onOpenChange,
  onDoDelete,
  isIrreversible = true,
  className,
}: {
  entityType: string;
  entityName?: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDoDelete: () => void;
  isIrreversible?: boolean;
  className?: string;
}): React.ReactNode {
  const displayType = entityType
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className={className}>
        <DialogHeader>
          <DialogTitle>
            Delete {displayType} {entityName && `"${entityName}"`}
          </DialogTitle>
          <DialogDescription>
            Are you sure you want to delete this {entityType}?
            {isIrreversible && (
              <b>
                <br /> This action cannot be undone.
              </b>
            )}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button variant="destructive" onClick={onDoDelete}>
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
