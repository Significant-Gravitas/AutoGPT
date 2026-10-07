import { useUserTimezone } from "@/lib/hooks/useUserTimezone";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

export function TimezoneNotice() {
  const userTimezone = useUserTimezone();

  if (!userTimezone) {
    return null;
  }

  if (userTimezone === "not-set") {
    return (
      <div className="mt-1 flex items-center gap-2 rounded-md border border-yellow-200 bg-yellow-50 p-3">
        <Icon
          icon={InformationCircleIcon}
          className="h-4 w-4 text-yellow-600"
        />
        <Text variant="body" className="text-yellow-800">
          No timezone set. Schedule will run in UTC.
          <a href="/settings/account" className="ml-1 underline">
            Set your timezone
          </a>
        </Text>
      </div>
    );
  }

  const tzName = getTimezoneDisplayName(userTimezone || "UTC");

  return (
    <div className="mt-1 flex items-center gap-2 rounded-md bg-zinc-100/50 p-3">
      <Icon
        icon={InformationCircleIcon}
        className="h-4 w-4 text-muted-foreground"
      />
      <Text variant="body" tone="muted">
        Schedule will run in your timezone:{" "}
        <span className="font-medium">{tzName}</span>
      </Text>
    </div>
  );
}
