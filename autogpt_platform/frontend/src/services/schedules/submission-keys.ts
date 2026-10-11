// Idempotency keys for schedule creates. Retrying the same payload reuses its
// key, so a create the scheduler applied without confirming isn't doubled; a
// changed payload is a new submission and gets a new key.
export function createSubmissionKeys() {
  let last: { payload: string; key: string } | null = null;

  return {
    keyFor(payload: unknown): string {
      const serialized = JSON.stringify(payload);
      if (last?.payload !== serialized) {
        last = { payload: serialized, key: crypto.randomUUID() };
      }
      return last.key;
    },
    reset() {
      last = null;
    },
  };
}
