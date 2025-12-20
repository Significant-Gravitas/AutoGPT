import React from "react";
import * as Sentry from "@sentry/nextjs";
import { redirect } from "next/navigation";
import { getServerUser } from "./auth/server/getServerAuth";

export async function withRoleAccess(allowedRoles: string[]) {
  "use server";
  return await Sentry.withServerActionInstrumentation(
    "withRoleAccess",
    {},
    async () => {
      return async function <T extends React.ComponentType<any>>(Component: T) {
        const user = await getServerUser();

        if (!user || !user.role || !allowedRoles.includes(user.role)) {
          redirect("/unauthorized");
        }
        return Component;
      };
    },
  );
}
