import { inject } from "vitest";

declare module "vitest" {
  export interface ProvidedContext {
    processEnvBeforeStorybook: Record<string, string>;
  }
}

// See vitest.config.mts: drop what Storybook's Next.js plugin loaded from
// .env, so unit tests see the environment the run started with.
const original = inject("processEnvBeforeStorybook");
for (const key of Object.keys(process.env)) {
  if (!(key in original)) delete process.env[key];
}
Object.assign(process.env, original);
