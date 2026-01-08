import { IconButtonProps, TranslatableString } from "@rjsf/utils";
import { cn } from "@/lib/utils";
import { Button } from "@/components/atoms/Button/Button";
import { PlusIcon } from "@phosphor-icons/react";

export default function AddButton({
  registry,
  className,
  uiSchema: _uiSchema,
  ...props
}: IconButtonProps) {
  const { translateString } = registry;
  return (
    <div className="m-0 w-full p-0">
      <Button
        {...props}
        size="small"
        className={cn("w-full gap-4", className)}
        variant="secondary"
        type="button"
      >
        <PlusIcon size={16} weight="bold" />
        {translateString(TranslatableString.AddItemButton)}
      </Button>
    </div>
  );
}
