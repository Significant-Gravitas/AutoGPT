"use client";

import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { useLayoutEffect } from "react";

interface Props {
  // The signed-in user the layout read on the server; null when it found no
  // session or couldn't read it, which leaves the store to its own check.
  user: User | null;
}

// Hands the server's user to the auth store before any effect of the page
// runs, so the page renders without waiting for the browser's session check.
// Not during render: on the server the store is shared by every request.
export function ServerUserSeed({ user }: Props) {
  useLayoutEffect(() => {
    if (user) useAuthStore.getState().seedFromServer(user);
  }, [user]);
  return null;
}
