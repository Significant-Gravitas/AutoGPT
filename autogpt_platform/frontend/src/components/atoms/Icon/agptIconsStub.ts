// Empty stand-in for the optional `@autogpt/icons` package.
//
// `@autogpt/icons` is a private package that self-hosters may not have access
// to, so it is declared as an optionalDependency. When it isn't installed, the
// bundler aliases `@autogpt/icons` to this module (see next.config.mjs) so the
// build never fails on a missing import. A `import * as icons` against this
// module yields an empty namespace, which makes every lookup fall back to the
// Phosphor icon.
export {};
