// What the public share reads answer for a dead link: 404 when the token was
// never shared, has been unshared or its run or chat is gone, and 422 when the
// token isn't shaped like a share token at all. Asking again won't change
// either answer.
const DEAD_LINK_STATUSES = [404, 422];

const MAX_SHARE_RETRIES = 3;

export function isDeadLinkError(error: unknown) {
  return (
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    typeof error.status === "number" &&
    DEAD_LINK_STATUSES.includes(error.status)
  );
}

export function retryUnlessDeadLink(failureCount: number, error: unknown) {
  return !isDeadLinkError(error) && failureCount < MAX_SHARE_RETRIES;
}
