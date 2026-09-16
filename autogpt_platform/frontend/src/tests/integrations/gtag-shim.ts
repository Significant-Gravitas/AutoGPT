// Stands in for the `gtag()` shim the Google tag init script defines on
// window, recording each command so tests can assert on what would reach the
// dataLayer.
export function installGtagShim(): unknown[][] {
  const calls: unknown[][] = [];
  window.gtag = (...args: unknown[]) => {
    calls.push(args);
  };
  return calls;
}

export function removeGtagShim() {
  delete window.gtag;
}
