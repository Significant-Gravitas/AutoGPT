/**
 * How long to wait for the LaunchDarkly client to initialise.
 *
 * The unit is **seconds**, not milliseconds: `LDProvider`'s `timeout` prop and
 * `LDClient.waitForInitialization()` both take seconds, and the JS SDK logs a
 * warning for anything above 5 ("we recommend a timeout of 5 seconds or less").
 * Sentry's captureConsoleIntegration turns that warning into an event on every
 * page load, so keep this at 5 or below.
 *
 * On timeout the SDK rejects, `LDProvider` swallows the rejection and
 * subscribes to `ready` instead, and the UI runs on the defaults in
 * `use-get-flag.ts` until LaunchDarkly answers.
 */
export const LD_INIT_TIMEOUT_SECONDS = 5;
