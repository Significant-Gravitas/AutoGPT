import { ErrorListProps, TranslatableString } from "@rjsf/utils";
import { AlertCircle } from "lucide-react";

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";

export default function ErrorList(props: ErrorListProps) {
  const { errors, registry } = props;
  const { translateString } = registry;
  return (
    <Alert variant="error" className="mb-2">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>{translateString(TranslatableString.ErrorsLabel)}</AlertTitle>
      <AlertDescription className="flex flex-col gap-1">
        {errors.map((error, i: number) => {
          return <span key={i}>&#x2022; {error.stack}</span>;
        })}
      </AlertDescription>
    </Alert>
  );
}
