import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { DeletedKey, getRerunFormNotice } from "../helpers";

interface Props {
  deletedKeys: DeletedKey[];
}

export function DeletedKeysNotice({ deletedKeys }: Props) {
  return (
    <Alert variant="warning">
      <AlertDescription>
        <Text variant="body">{getRerunFormNotice(deletedKeys)}</Text>
      </AlertDescription>
    </Alert>
  );
}
