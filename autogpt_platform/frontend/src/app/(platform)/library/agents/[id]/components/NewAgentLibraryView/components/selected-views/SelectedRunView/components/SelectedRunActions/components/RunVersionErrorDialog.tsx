import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DeletedKey,
  describeDeletedKeys,
  getRunVersionErrorNotice,
  RERUN_FORM_TITLE,
} from "../helpers";

interface Props {
  version: number;
  deletedKeys: DeletedKey[];
  isRetrying: boolean;
  onRetry: () => void;
  onOpenChange: (open: boolean) => void;
}

// Stands in for the rerun form while the workflow version the task ran can't
// be read: the form is only built from that version, never the current one
export function RunVersionErrorDialog({
  version,
  deletedKeys,
  isRetrying,
  onRetry,
  onOpenChange,
}: Props) {
  return (
    <Dialog
      title={RERUN_FORM_TITLE}
      controlled={{ isOpen: true, set: onOpenChange }}
      styling={{ maxWidth: "600px" }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Alert variant="warning">
            <AlertDescription>
              <Text variant="body">{describeDeletedKeys(deletedKeys)}</Text>
            </AlertDescription>
          </Alert>
          <Alert variant="error" aria-live="polite">
            <AlertDescription>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <Text variant="body">{getRunVersionErrorNotice(version)}</Text>
                <Button
                  variant="secondary"
                  size="small"
                  loading={isRetrying}
                  disabled={isRetrying}
                  onClick={onRetry}
                >
                  Try again
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
