import { Text } from "@/components/atoms/Text/Text";

type Props = {
  title: string;
  description: string;
  // Kept for API compatibility; rendering is owned by parent StepStrip now.
  currentStep?: "select" | "info" | "review";
};

export function StepHeader({ title, description }: Props) {
  return (
    <div className="flex max-w-[640px] flex-col gap-1 px-1 pb-6 sm:px-2">
      <Text variant="body" as="h2" className="text-textBlack">
        {title}
      </Text>
      <Text variant="small" className="text-zinc-600">
        {description}
      </Text>
    </div>
  );
}
