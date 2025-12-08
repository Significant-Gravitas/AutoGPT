import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}

export function ModalSection({ title, subtitle, children }: Props) {
  return (
    <div className="rounded-medium border border-zinc-200 p-6">
      <div className="mb-4 flex flex-col gap-1 border-b border-zinc-100 pb-4">
        <Text variant="lead-semibold">{title}</Text>
        <Text variant="body" className="text-zinc-500">
          {subtitle}
        </Text>
      </div>
      {children}
    </div>
  );
}
