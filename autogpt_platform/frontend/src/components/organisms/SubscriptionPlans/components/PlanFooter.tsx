import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onCompare: () => void;
}

export function PlanFooter({ onCompare }: Props) {
  return (
    <footer className="mt-3 flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
      <Text variant="small" tone="muted">
        Need something custom?{" "}
        <a
          href="mailto:sales@agpt.co"
          className="font-medium text-purple-500 no-underline hover:text-purple-600"
        >
          Talk to sales.
        </a>
      </Text>
      <Button
        type="button"
        variant="link"
        className="h-auto min-w-0 p-0 text-xs text-zinc-500"
        onClick={onCompare}
      >
        Compare plans
      </Button>
    </footer>
  );
}
