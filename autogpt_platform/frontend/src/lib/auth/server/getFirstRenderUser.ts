import type { User } from "../types";
import { getServerUser } from "./getServerUser";

// The layouts that seed the auth store wait for this read before they render,
// so it gives up after this long. The browser then checks the session itself,
// as it does when the server finds none.
export const FIRST_RENDER_SESSION_DEADLINE_MS = 1_500;

/**
 * The signed-in user for the page's first render, or `null` when there is no
 * session, the read failed, or it didn't answer within the deadline.
 */
export async function getFirstRenderUser(): Promise<User | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<null>((resolve) => {
    timer = setTimeout(() => {
      console.warn(
        "Session read missed the first-render deadline; the browser checks it instead",
      );
      resolve(null);
    }, FIRST_RENDER_SESSION_DEADLINE_MS);
  });
  try {
    return await Promise.race([
      getServerUser().then(({ user }) => user),
      deadline,
    ]);
  } finally {
    clearTimeout(timer);
  }
}
