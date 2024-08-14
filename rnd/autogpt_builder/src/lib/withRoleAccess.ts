import { redirect } from "next/navigation";
import getServerUser from "@/hooks/getServerUser";

export function withRoleAccess(allowedRoles: string[]) {
  return async (Component: React.ComponentType) => {
    const { user, role, error } = await getServerUser();

    if (error || !user || !role || !allowedRoles.includes(role)) {
      redirect("/unauthorized");
    }

    return Component;
  };
}
