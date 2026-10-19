import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import { SessionKey } from "@/services/storage/session-storage";

const DEPLOYMENT_RELOAD_COOLDOWN_MS = 60_000;

let hasPromptedForRefresh = false;

/**
 * Reloads the page so it picks up the deployment the server is now running.
 *
 * A timestamp in sessionStorage stops a server that keeps rejecting this
 * bundle from trapping the tab in a reload loop. When that guard can't be
 * kept across a reload (storage blocked) or it has already fired, the user is
 * asked to refresh instead. Returns whether the page is reloading.
 */
export function reloadForNewDeployment(
  reload: () => void = () => window.location.reload(),
): boolean {
  if (environment.isServerSide()) return false;

  const lastReloadAt = Number(readReloadGuard() ?? 0);
  const coolingDown = Date.now() - lastReloadAt < DEPLOYMENT_RELOAD_COOLDOWN_MS;
  if (!coolingDown && writeReloadGuard(Date.now())) {
    reload();
    return true;
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

function writeReloadGuard(at: number): boolean {
  try {
    window.sessionStorage.setItem(SessionKey.DEPLOYMENT_RELOAD_AT, String(at));
    return (
      window.sessionStorage.getItem(SessionKey.DEPLOYMENT_RELOAD_AT) ===
      String(at)
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
