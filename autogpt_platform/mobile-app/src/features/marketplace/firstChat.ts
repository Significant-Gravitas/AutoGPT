// No app imports, so its test loads without native modules.

export const FIRST_CHAT_POLLS = 15;
export const FIRST_CHAT_POLL_MS = 1_000;

interface Steps {
  /** The expert's newest chat id, if any. */
  findLatest: () => Promise<string | null>;
  create: () => Promise<string>;
  wait: (ms: number) => Promise<void>;
  polls?: number;
}

/**
 * The chat a just-hired expert opens in. The server starts that chat and its
 * hello in the background, and skips it once any chat exists, so creating one
 * first would leave the expert silent. Wait for the server's; make one only if
 * it never comes.
 */
export async function awaitFirstChat({
  findLatest,
  create,
  wait,
  polls = FIRST_CHAT_POLLS,
}: Steps) {
  for (let attempt = 0; attempt < polls; attempt += 1) {
    const latest = await findLatest();
    if (latest) return latest;
    if (attempt < polls - 1) await wait(FIRST_CHAT_POLL_MS);
  }
  return create();
}
