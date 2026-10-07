import { Text } from "@/components/atoms/Text/Text";
import { OrbitLoader } from "../../../OrbitLoader/OrbitLoader";

interface Props {
  expertName?: string;
}

export function ExpertKickoffLoader({ expertName }: Props) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex h-full flex-1 flex-col items-center justify-center gap-4 text-center"
    >
      <OrbitLoader size={32} />
      <div>
        <Text variant="large-medium" tone="primary" as="p" unmask={false}>
          {expertName
            ? `Opening ${expertName}'s workspace`
            : "Opening workspace"}
        </Text>
        <Text variant="body" tone="muted" as="p" className="mt-1">
          Your expert is getting ready to start.
        </Text>
      </div>
    </div>
  );
}
