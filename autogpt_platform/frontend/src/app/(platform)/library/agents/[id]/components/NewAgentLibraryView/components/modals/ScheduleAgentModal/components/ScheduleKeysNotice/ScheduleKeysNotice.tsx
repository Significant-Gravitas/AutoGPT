import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { ReplacedKey } from "../../helpers";

interface Props {
  replacedKeys: ReplacedKey[];
  usesFormKeys: boolean;
  onUseFormKeys: () => void;
}

export function ScheduleKeysNotice({
  replacedKeys,
  usesFormKeys,
  onUseFormKeys,
}: Props) {
  if (replacedKeys.length === 0) return null;

  if (usesFormKeys) {
    return (
      <div className="mt-4">
        <Alert>
          <AlertDescription>
            <Text variant="body">
              This schedule uses the keys shown in the run form.
            </Text>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="mt-4">
      <Alert variant="warning">
        <AlertDescription>
          <div className="flex flex-col items-start gap-2">
            <Text variant="body">
              This schedule uses different keys from the run form:
            </Text>
            <ul className="list-disc space-y-1 pl-5">
              {replacedKeys.map(({ field, text }) => (
                <Text key={field} as="li" variant="body">
                  {text}
                </Text>
              ))}
            </ul>
            <Button variant="secondary" size="small" onClick={onUseFormKeys}>
              Use the keys shown in the run form
            </Button>
          </div>
        </AlertDescription>
      </Alert>
    </div>
  );
}
