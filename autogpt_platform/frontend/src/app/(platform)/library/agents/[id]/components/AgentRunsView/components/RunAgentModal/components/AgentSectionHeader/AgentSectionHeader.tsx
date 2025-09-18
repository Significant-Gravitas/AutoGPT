import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
}

export function AgentSectionHeader({ title }: Props) {
  return (
    <div className="border-t border-zinc-400 px-0 pb-2 pt-1">
      <Text variant="label" className="!text-zinc-700">
        {title}
      </Text>
    </div>
  );
}
