import React from "react";
import * as Sentry from "@sentry/nextjs";

export async function withRoleAccess(allowedRoles: string[]) {
  console.log("withRoleAccess called:", allowedRoles);
  ("use server");
  return await Sentry.withServerActionInstrumentation(
    "withRoleAccess",
    {},
    async () => {
      return async function <T extends React.ComponentType<any>>(Component: T) {
        return Component;
      };
    },
  );
}
