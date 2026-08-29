import { Text } from "@/components/atoms/Text/Text";

export function AuthDivider() {
  return (
    <div className="my-5 flex w-full items-center gap-3">
      <span className="h-px flex-1 bg-slate-200" />
      <Text variant="small-medium" className="!text-slate-500">
        or
      </Text>
      <span className="h-px flex-1 bg-slate-200" />
    </div>
  );
}
