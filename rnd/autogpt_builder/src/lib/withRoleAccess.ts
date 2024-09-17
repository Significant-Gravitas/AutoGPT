import { redirect } from "next/navigation";
import getServerUser from "@/hooks/getServerUser";
import React from "react";
import * as Sentry from "@sentry/nextjs";

export async function withRoleAccess(allowedRoles: string[]) {
  "use server";
  return await Sentry.withServerActionInstrumentation(
    "withRoleAccess",
    {},
    async () => {
      return async function <T extends React.ComponentType<any>>(Component: T) {
        const { user, role, error } = await getServerUser();

        if (error || !user || !role || !allowedRoles.includes(role)) {
          redirect("/unauthorized");
        }

        return Component;
      };
    },
  );
}
