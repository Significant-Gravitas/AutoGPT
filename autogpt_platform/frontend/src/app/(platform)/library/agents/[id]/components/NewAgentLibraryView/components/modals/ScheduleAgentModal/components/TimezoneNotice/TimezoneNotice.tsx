import { useUserTimezone } from "@/lib/hooks/useUserTimezone";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { InfoIcon } from "@phosphor-icons/react";

export function TimezoneNotice() {
  const userTimezone = useUserTimezone();

  if (!userTimezone) {
    return null;
  }

  if (userTimezone === "not-set") {
    return (
      <div className="mt-1 flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 p-3">
        <InfoIcon className="h-4 w-4 text-amber-600" />
        <p className="text-sm text-amber-800">
          No timezone set. Schedule will run in UTC.
          <a href="/profile/settings" className="ml-1 underline">
            Set your timezone
          </a>
        </p>
      </div>
    );
  }

  const tzName = getTimezoneDisplayName(userTimezone || "UTC");

  return (
    <div className="mt-1 flex items-center gap-2 rounded-md bg-muted/50 p-3">
      <InfoIcon className="h-4 w-4 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">
        Schedule will run in your timezone:{" "}
        <span className="font-medium">{tzName}</span>
      </p>
    </div>
  );
}
