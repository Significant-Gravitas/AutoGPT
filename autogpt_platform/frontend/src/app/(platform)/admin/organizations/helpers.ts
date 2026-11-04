export const PAGE_SIZE = 25;

export const ACCOUNT_QUERY_PARAM = "account";

export function getTeamPlansHref(email: string) {
  return `/admin/organizations?${new URLSearchParams({ [ACCOUNT_QUERY_PARAM]: email })}`;
}

export function formatGrantDate(value: string | Date) {
  return new Date(value).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong.";
}

export function pageCount(total: number, pageSize: number) {
  return Math.max(1, Math.ceil(total / pageSize));
}
