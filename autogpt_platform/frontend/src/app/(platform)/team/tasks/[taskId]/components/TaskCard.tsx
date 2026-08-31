import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  children: React.ReactNode;
}

/** One card in the task detail's right rail. */
export function TaskCard({ title, children }: Props) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl bg-white p-4 ring-[0.5px] ring-zinc-200">
      <Text variant="small" className="font-medium text-zinc-900">
        {title}
      </Text>
      {children}
    </section>
  );
}
