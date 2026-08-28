import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

const TECHNICAL_NAMES = new Set(["agent.json", "build_state.json"]);
const SDK_RESULT = /^sdk-[0-9a-f-]{8,}\.json$/i;
const INTERNAL_ROLES = new Set(["diagnostic", "debug", "build"]);

function metadataString(file: WorkspaceFileItem, key: string) {
  const value = file.metadata?.[key];
  return typeof value === "string" ? value.trim() : "";
}

export function isTechnicalWorkspaceFile(file: WorkspaceFileItem) {
  const name = file.name.trim().toLowerCase();
  const path = file.path.toLowerCase();
  const generated = file.origin === "generated";
  return (
    metadataString(file, "audience").toLowerCase() === "internal" ||
    INTERNAL_ROLES.has(metadataString(file, "artifact_role").toLowerCase()) ||
    (generated &&
      (TECHNICAL_NAMES.has(name) ||
        SDK_RESULT.test(name) ||
        path.includes("/tool-outputs/")))
  );
}

export function workspaceFileTitle(file: WorkspaceFileItem) {
  return metadataString(file, "title") || file.name;
}

export function workspaceFileOwner(file: WorkspaceFileItem) {
  return metadataString(file, "owner_name");
}

export function workspaceFilePurpose(file: WorkspaceFileItem) {
  return metadataString(file, "purpose");
}

export function workspaceFileVerification(file: WorkspaceFileItem) {
  const verification = metadataString(file, "verification").toLowerCase();
  if (verification === "verified") return "Verified";
  if (verification === "likely") return "Likely";
  if (verification === "disqualified") return "Disqualified";
  return verification ? "Unknown" : "";
}
