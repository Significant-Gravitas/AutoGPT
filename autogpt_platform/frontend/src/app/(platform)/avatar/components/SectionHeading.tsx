import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  hint: string;
}

export function SectionHeading({ title, hint }: Props) {
  return (
    <div className="flex items-baseline gap-3">
      <Text variant="body-medium" as="h2" className="text-zinc-900">
        {title}
      </Text>
      <Text variant="small" as="p" className="text-zinc-500">
        {hint}
      </Text>
    </div>
  );
}
