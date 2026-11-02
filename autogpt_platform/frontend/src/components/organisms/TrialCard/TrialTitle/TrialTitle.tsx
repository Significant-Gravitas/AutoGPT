import { Text } from "@/components/atoms/Text/Text";

interface Props {
  children: React.ReactNode;
}

export function TrialTitle({ children }: Props) {
  return (
    <Text variant="h5-semibold" as="h2" tone="emphasis">
      {children}
    </Text>
  );
}
