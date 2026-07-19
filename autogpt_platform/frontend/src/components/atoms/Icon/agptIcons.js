// Runtime bridge to the optional `@autogpt/icons` package.
//
// This is intentionally a .js file (not .ts): `@autogpt/icons` is a private
// optionalDependency that self-hosters won't have installed, so the bare import
// below can't be type-checked in that scenario. With `allowJs: false`, tsc
// skips this file entirely — while the bundler still processes it (aliasing the
// import to a local empty stub when the package is absent; see next.config.mjs).
// Consumers get full types from the colocated agptIcons.d.ts. This keeps the
// optional import out of the type graph without a `@ts-ignore` suppressor.
import * as autogptIcons from "@autogpt/icons";

// Returns the AutoGPT icon component for the given export name, or undefined
// when the package isn't installed (stub) or doesn't export that icon.
export function getAutoGPTIcon(exportName) {
  return autogptIcons[exportName];
}
