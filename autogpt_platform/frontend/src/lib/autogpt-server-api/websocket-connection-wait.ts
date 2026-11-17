export async function waitForWebSocketConnection(
  connection: Promise<void>,
): Promise<void> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      connection,
      new Promise<never>((_, reject) => {
        timeout = setTimeout(
          () => reject(new Error("WebSocket connection timed out")),
          30_000,
        );
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}
