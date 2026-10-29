import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import { SessionKey } from "@/services/storage/session-storage";

const DEPLOYMENT_RELOAD_COOLDOWN_MS = 60_000;
// Stored in place of the reload's timestamp once the page it loaded was
// rejected too: reloading can't fix that, so it isn't tried again in this tab.
const RELOAD_DID_NOT_HELP = "did-not-help";

let hasPromptedForRefresh = false;

/**
 * Reloads the page so it picks up the deployment the server is now running.
 *
 * A guard in sessionStorage stops a server that keeps rejecting this bundle
 * from trapping the tab in a reload loop: a rejection within the cooldown
 * means the reload didn't help, and from then on the user is asked to refresh
 * instead. They're also asked when the guard can't be kept across a reload
 * (storage blocked). Returns whether the page is reloading.
 */
export function reloadForNewDeployment(
  reload: () => void = () => window.location.reload(),
): boolean {
  if (environment.isServerSide()) return false;

  const guard = readReloadGuard();
  if (guard !== RELOAD_DID_NOT_HELP) {
    const coolingDown =
      Date.now() - Number(guard ?? 0) < DEPLOYMENT_RELOAD_COOLDOWN_MS;
    if (coolingDown) {
      writeReloadGuard(RELOAD_DID_NOT_HELP);
    } else if (writeReloadGuard(String(Date.now()))) {
      reload();
      return true;
    }
  }

  promptForRefresh();
  return false;
}

// The guard goes straight to the browser's sessionStorage, not the safe
// wrapper: the wrapper falls back to memory, which the reload itself wipes.
function readReloadGuard(): string | null {
  try {
    return window.sessionStorage.getItem(SessionKey.DEPLOYMENT_RELOAD_AT);
  } catch {
    return null;
  }
}

function writeReloadGuard(value: string): boolean {
  try {
    window.sessionStorage.setItem(SessionKey.DEPLOYMENT_RELOAD_AT, value);
    return (
      window.sessionStorage.getItem(SessionKey.DEPLOYMENT_RELOAD_AT) === value
    );
  } catch {
    return false;
  }
}

function promptForRefresh(): void {
  if (hasPromptedForRefresh) return;
  hasPromptedForRefresh = true;
  toast({
    title: "A new version of AutoGPT is available",
    description: "Refresh the page to keep going.",
    duration: Infinity,
  });
}
