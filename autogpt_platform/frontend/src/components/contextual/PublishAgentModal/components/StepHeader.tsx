import { Text } from "@/components/atoms/Text/Text";

type Props = {
  title: string;
  description: string;
};

export function StepHeader({ title, description }: Props) {
  return (
    <div className="relative border-b border-neutral-200 px-4 pb-4 sm:px-6 sm:pb-6">
      <div className="text-center">
        <Text variant="h3">{title}</Text>
        <Text variant="large">{description}</Text>
      </div>
    </div>
  );
}
