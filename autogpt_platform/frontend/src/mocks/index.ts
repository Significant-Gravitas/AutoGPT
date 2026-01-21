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
