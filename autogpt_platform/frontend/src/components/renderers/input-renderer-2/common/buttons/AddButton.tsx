import {
  FormContextType,
  IconButtonProps,
  RJSFSchema,
  StrictRJSFSchema,
  TranslatableString,
} from "@rjsf/utils";
import { cn } from "../lib/utils";
import { Button } from "@/components/atoms/Button/Button";
import { PlusCircleIcon } from "@phosphor-icons/react";

export default function AddButton<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({ uiSchema, registry, className, ...props }: IconButtonProps<T, S, F>) {
  const { translateString } = registry;
  return (
    <div className="m-0 p-0">
      <Button
        {...props}
        size="small"
        className={cn("w-fit gap-2", className)}
        variant="outline"
        type="button"
      >
        <PlusCircleIcon size={20} />
        {translateString(TranslatableString.AddItemButton)}
      </Button>
    </div>
  );
}
