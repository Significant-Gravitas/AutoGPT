import { Button } from "@/components/atoms/Button/Button";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

const UNCONFIRMED =
  "We couldn't confirm the schedule was saved, so it may already exist. Check your schedules before trying again.";
const EARLIER_TRY =
  "A schedule from an earlier try may already exist. Check your schedules before trying again.";

// With no answer, a server error or a 409 for the create key, the schedule
// may already exist. Refresh the lists the toast sends the user to check.
export function describeScheduleCreateError(error: unknown) {
  if (!(error instanceof ApiError) || error.status >= 500) {
    return {
      shouldRefresh: true,
      toast: { description: UNCONFIRMED, action: <ViewSchedulesButton /> },
    };
  }
  if (error.status === 409) {
    return {
      shouldRefresh: true,
      toast: { description: EARLIER_TRY, action: <ViewSchedulesButton /> },
    };
  }
  return {
    shouldRefresh: false,
    toast: { description: error.message || "An unexpected error occurred." },
  };
}

function ViewSchedulesButton() {
  return (
    <Button
      as="NextLink"
      href="/library/followups"
      variant="secondary"
      size="small"
    >
      View schedules
    </Button>
  );
}
