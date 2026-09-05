import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";

export function EmptyTeamState() {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-zinc-300 bg-white p-8 text-center">
      <Icon icon={UserGroupIcon} size={32} className="text-zinc-400" />
      <Text variant="large-medium">No hired experts yet</Text>
      <Text variant="body" className="max-w-prose text-zinc-600">
        Hire an expert from the marketplace and they will show up here, ready to
        work alongside Autopilot.
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button
          as="NextLink"
          href="/marketplace"
          variant="primary"
          size="small"
          className={ACTION_BUTTON_CLASS}
        >
          Browse the marketplace
        </Button>
        <Button
          as="NextLink"
          href="/raise"
          variant="secondary"
          size="small"
          className={ACTION_BUTTON_CLASS}
        >
          Raise your own
        </Button>
      </div>
    </div>
  );
}
