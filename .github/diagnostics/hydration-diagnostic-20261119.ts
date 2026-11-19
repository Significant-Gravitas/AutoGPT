import type { Page, TestInfo } from "@playwright/test";

const describeFiber = `function () {
  if (!this || typeof this.tag !== "number" || !("return" in this)) return null;
  const frames = [];
  for (let fiber = this; fiber && frames.length < 35; fiber = fiber.return) {
    const type = fiber.type;
    const props = fiber.pendingProps;
    const node = fiber.stateNode;
    frames.push({
      tag: fiber.tag,
      type: typeof type === "string" ? type : type?.displayName || type?.name || typeof type,
      source: typeof type === "function" ? Function.prototype.toString.call(type).slice(0, 350) : null,
      props: props && typeof props === "object" ? Object.keys(props) : [],
      className: typeof props?.className === "string" ? props.className : null,
      testId: props?.["data-testid"],
      node: node?.nodeType === 1 ? { tag: node.tagName, id: node.id, className: node.getAttribute("class") } : null,
    });
  }
  return frames;
}`;

export async function captureHydration(page: Page, testInfo: TestInfo) {
  const session = await page.context().newCDPSession(page);
  const records: unknown[] = [];
  const pending = new Set<Promise<void>>();
  await session.send("Debugger.enable");
  session.on("Debugger.paused", (event) => {
    const task = (async () => {
      try {
        const description = event.data?.description ?? "";
        if (!/React error #(418|519)|[Hh]ydration/.test(description)) return;
        const fibers: unknown[] = [];
        for (const frame of event.callFrames.slice(0, 3)) {
          const scope = frame.scopeChain.find((item) => item.type === "local");
          if (!scope?.object.objectId) continue;
          const properties = await session.send("Runtime.getProperties", {
            objectId: scope.object.objectId,
            ownProperties: true,
          });
          for (const property of properties.result) {
            if (property.value?.type !== "object" || !property.value.objectId)
              continue;
            const result = await session.send("Runtime.callFunctionOn", {
              objectId: property.value.objectId,
              functionDeclaration: describeFiber,
              returnByValue: true,
              silent: true,
            });
            if (result.result.value) {
              fibers.push({
                function: frame.functionName,
                variable: property.name,
                frames: result.result.value,
              });
            }
          }
        }
        records.push({ description, fibers });
      } catch (error) {
        records.push({ diagnosticError: String(error) });
      } finally {
        await session.send("Debugger.resume").catch(() => {});
      }
    })();
    pending.add(task);
    void task.finally(() => pending.delete(task));
  });
  await session.send("Debugger.setPauseOnExceptions", { state: "all" });
  return async function finish() {
    await session.send("Debugger.setPauseOnExceptions", { state: "none" });
    await Promise.all(pending);
    await session.detach();
    await testInfo.attach("hydration-diagnostics", {
      body: JSON.stringify(records, null, 2),
      contentType: "application/json",
    });
  };
}
