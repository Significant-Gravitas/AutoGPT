import { TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { Key, storage } from "@/services/storage/local-storage";

// Sentinel Select value for the "Organization" (org-home) choice. The Select
// atom only handles string values, so null teamId maps to this sentinel.
export const ORG_HOME_OPTION_VALUE = "__org_home__";

// Stable per-surface keys for last-used-team storage. One key per distinct
// create surface so each remembers its own default.
export const CreateSurface = {
  BuilderSave: "builder-save",
  BuilderDuplicate: "builder-duplicate",
  BuilderSchedule: "builder-schedule",
  LibraryFolder: "library-folder",
  LibraryUpload: "library-upload",
  ScheduleAgent: "schedule-agent",
  ApiKey: "api-key",
} as const;

// Stored marker for "org-home was the last-used choice on this surface".
const ORG_HOME_STORED = "org-home";

function readMap(): Record<string, string> {
  const raw = storage.get(Key.CREATE_SURFACE_TEAMS);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

// Last team the user picked on a given create surface (null == org-home, or
// never used). Callers seed the picker's default from this.
export function getLastUsedTeam(surfaceKey: string): string | null {
  const value = readMap()[surfaceKey];
  return value && value !== ORG_HOME_STORED ? value : null;
}

export function setLastUsedTeam(
  surfaceKey: string,
  teamId: string | null,
): void {
  const map = readMap();
  map[surfaceKey] = teamId ?? ORG_HOME_STORED;
  storage.set(Key.CREATE_SURFACE_TEAMS, JSON.stringify(map));
}

// Per-request Orval options (second arg to customMutator) that stamp the
// chosen team via the X-Team-Id header. Returns undefined for org-home so the
// backend falls back to the active-org context.
export function getTeamRequestInit(
  teamId: string | null,
): RequestInit | undefined {
  if (!teamId) return undefined;
  return { headers: { [TEAM_HEADER_NAME]: teamId } };
}
