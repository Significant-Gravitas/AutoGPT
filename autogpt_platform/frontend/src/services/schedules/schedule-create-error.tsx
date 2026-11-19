import { Button } from "@/components/atoms/Button/Button";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

const UNCONFIRMED =
  "We couldn't confirm the schedule was saved, so it may already exist. Check your schedules before trying again.";
const EARLIER_TRY =
  "A schedule from an earlier try may already exist. Check your schedules before trying again.";

// The toast for a failed schedule create. With no answer, a server error or
// a 409 for its key, the schedule may exist already, so the toast points to
// the schedules instead of inviting a retry that could add a second one.
export function scheduleCreateErrorToast(error: unknown) {
  if (!(error instanceof ApiError) || error.status >= 500) {
    return { description: UNCONFIRMED, action: <ViewSchedulesButton /> };
  }
  if (error.status === 409) {
    return { description: EARLIER_TRY, action: <ViewSchedulesButton /> };
  }
  return { description: error.message || "An unexpected error occurred." };
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
