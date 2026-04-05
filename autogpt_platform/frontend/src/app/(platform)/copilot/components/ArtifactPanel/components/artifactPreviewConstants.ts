// Re-export shared iframe sandbox constants. Kept as a local barrel so
// existing imports inside the copilot feature don't have to change.
export {
  ARTIFACT_IFRAME_CSP,
  TAILWIND_CDN_URL,
} from "@/lib/iframe-sandbox-csp";
