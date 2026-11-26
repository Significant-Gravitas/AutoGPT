import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { useId } from "react";

interface Props {
  label: string;
  description: string;
  checked: boolean;
  disabled: boolean;
  onCheckedChange: () => void;
}

export function SafeModeSetting({
  label,
  description,
  checked,
  disabled,
  onCheckedChange,
}: Props) {
  const switchID = useId();
  const descriptionID = useId();

  return (
    <div className="flex w-full flex-col items-start gap-4 rounded-xl border border-separator bg-surface p-6">
      <div className="flex w-full items-start justify-between gap-4">
        <div className="flex-1">
          <Text
            variant="large"
            weight="semibold"
            as="label"
            htmlFor={switchID}
            className="block"
          >
            {label}
          </Text>
          <Text
            id={descriptionID}
            variant="large"
            tone="primary"
            className="mt-1"
          >
            {description}
          </Text>
        </div>
        <Switch
          id={switchID}
          aria-describedby={descriptionID}
          checked={checked}
          onCheckedChange={onCheckedChange}
          disabled={disabled}
          className="mt-1"
        />
      </div>
    </div>
  );
}
