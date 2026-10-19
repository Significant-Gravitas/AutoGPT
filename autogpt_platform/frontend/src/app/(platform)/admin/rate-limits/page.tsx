import { redirect } from "next/navigation";
import { getAdminUsersHref } from "../users/helpers";

interface SearchParams {
  user?: string;
  user_id?: string;
  email?: string;
}

export default async function RateLimitsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const { user, user_id, email } = await searchParams;
  redirect(getAdminUsersHref(user ?? user_id ?? email));
}
