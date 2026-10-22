import { ImpersonationState } from "@/lib/impersonation";
import { redirect } from "next/navigation";
import { getAdminUsersHref } from "../users/helpers";

interface SearchParams {
  user?: string | string[];
  user_id?: string | string[];
}

export default async function ImpersonationPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const { user, user_id } = await searchParams;
  const impersonatedUserId = await ImpersonationState.getServerSide();
  redirect(getAdminUsersHref(user ?? user_id ?? impersonatedUserId));
}
