import { Text } from "@/components/atoms/Text/Text";

interface Props {
  children: React.ReactNode;
}

/** A quiet caption that splits a panel into sections without a second
 *  header row. */
export function HomeSectionLabel({ children }: Props) {
  return (
    <Text
      variant="small-medium"
      className="px-4 pb-1 pt-3 text-[11px] uppercase tracking-[0.06em] text-zinc-400"
    >
      {children}
    </Text>
  );
}
