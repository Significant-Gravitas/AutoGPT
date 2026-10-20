import { Text } from "@/components/atoms/Text/Text";

interface Props {
  children: React.ReactNode;
}

/** A quiet caption that splits a panel into sections without a second
 *  header row. */
export function HomeSectionLabel({ children }: Props) {
  return (
    <Text variant="eyebrow" className="block px-4 pb-1 pt-3">
      {children}
    </Text>
  );
}
