import { environment } from "@/services/environment";

// The base properties every PostHog event carries (see
// docs/engineering/tracking-plan.md), registered once as super properties so
// no call site has to repeat them. `source` names the emitter; the backend
// sends `platform` or `chat_copilot`. Vercel branch previews share the
// production NEXT_PUBLIC_APP_ENV, so they are told apart here, as in
// `environment.getEnvironmentStr`. A custom Vercel environment, such as the dev
// deployment's `staging`, is not a preview and reports its own app env.
// `revision` is the build's commit, left out of a build that has none.
export function getPostHogBaseProperties() {
  const revision = environment.getRevision();
  return {
    source: "web",
    environment: environment.isVercelPreview()
      ? "preview"
      : environment.getAppEnv(),
    ...(revision ? { revision } : {}),
  };
}
