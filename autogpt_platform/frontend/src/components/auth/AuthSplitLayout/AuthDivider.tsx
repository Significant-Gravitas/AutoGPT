import { Text } from "@/components/atoms/Text/Text";

export function AuthDivider() {
  return (
    <div className="my-5 flex w-full items-center gap-3">
      <span className="h-px flex-1 bg-surface-active" />
      <Text variant="small-medium" className="!text-foreground-tertiary">
        or
      </Text>
      <span className="h-px flex-1 bg-surface-active" />
    </div>
  );
}
