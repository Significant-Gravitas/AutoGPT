import { IconButtonProps, TranslatableString } from "@rjsf/utils";
import { cn } from "@/lib/utils";
import { Button } from "@/components/atoms/Button/Button";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
        <Icon icon={PlusSignIcon} size={16} />
        {translateString(TranslatableString.AddItemButton)}
      </Button>
    </div>
  );
}
