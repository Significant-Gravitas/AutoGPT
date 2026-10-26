import { Text } from "@/components/atoms/Text/Text";

interface Props {
  children: React.ReactNode;
}

export function TrialTitle({ children }: Props) {
  return (
    <Text
      variant="h5"
      as="h2"
      // TODO(tokens): SECRT-3152, the 17px size has no role.
      className="font-poppins !text-[17px] !font-semibold !text-foreground-emphasis"
    >
      {children}
    </Text>
  );
}
