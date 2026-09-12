import { Text } from "@/components/atoms/Text/Text";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";

interface Props {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled: boolean;
}

export function MCPWriteAccessField({ checked, onChange, disabled }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <MultiToggle
        items={[{ value: "write", label: "Allow changes", disabled }]}
        selectedValues={checked ? ["write"] : []}
        onChange={(values) => onChange(values.includes("write"))}
      />
      <Text variant="small" className="text-zinc-600">
        Adds the optional changes described in this service&apos;s setup
        instructions.
      </Text>
    </div>
  );
}
