import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function EmptyTeamState() {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-zinc-300 bg-white p-8 text-center">
      <Icon icon={UserGroupIcon} size={32} className="text-zinc-400" />
      <Text variant="large-medium" tone="primary">
        No hired experts yet
      </Text>
      <Text variant="body" tone="secondary" className="max-w-prose">
        Hire an expert from the marketplace and they will show up here, ready to
        work alongside Autopilot.
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button as="NextLink" href="/marketplace" variant="primary" size="xs">
          Browse the marketplace
        </Button>
        <Button as="NextLink" href="/raise" variant="secondary" size="xs">
          Raise your own
        </Button>
      </div>
    </div>
  );
}
