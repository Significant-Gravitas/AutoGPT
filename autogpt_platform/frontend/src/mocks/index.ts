// We are not using this for tests because Vitest runs our tests in a Node.js environment.
// However, we can use it for development purposes to test our UI in the browser with fake data.
export async function initMocks() {
  if (typeof window === "undefined") {
    const { server } = await import("./mock-server");
    server.listen({ onUnhandledRequest: "bypass" });
    console.log("[MSW] Server mock initialized");
  } else {
    const { worker } = await import("./mock-browser");
    await worker.start({ onUnhandledRequest: "bypass" });
    console.log("[MSW] Browser mock initialized");
  }
}
