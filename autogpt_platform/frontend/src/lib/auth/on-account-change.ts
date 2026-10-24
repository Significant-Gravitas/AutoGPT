import { useAuthStore } from "./hooks/useAuthStore";

// Runs `reset` the moment a different account signs in than the one this
// page last had, for client state that belongs to an account. A sign-out
// alone does not count: a failed session check also drops the user, and the
// same person signing back in keeps their place. It runs synchronously
// inside the sign-in itself, so nothing the next account starts can race it.
export function onAccountChange(reset: () => void) {
  let account = useAuthStore.getState().user?.id ?? null;
  return useAuthStore.subscribe((state) => {
    const id = state.user?.id ?? null;
    if (id === null || id === account) return;
    if (account !== null) reset();
    account = id;
  });
}
