import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  children: React.ReactNode;
}

export function DetailSection({ title, children }: Props) {
  return (
    <section className="flex flex-col gap-1.5">
      <Text variant="eyebrow">{title}</Text>
      {children}
    </section>
  );
}
