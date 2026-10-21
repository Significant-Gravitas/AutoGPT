import type { Locator, Page } from "@playwright/test";
import { describe, expect, it, vi } from "vitest";

import { dismissTabIntros } from "./onboarding";

type ButtonHandle = Awaited<ReturnType<Locator["elementHandles"]>>[number];

function createButton() {
  return {
    press: vi.fn<ButtonHandle["press"]>().mockResolvedValue(undefined),
    dispose: vi.fn<ButtonHandle["dispose"]>().mockResolvedValue(undefined),
  };
}

async function registerIntroHandler() {
  const elementHandles = vi.fn<Locator["elementHandles"]>();
  const intro = {
    getByRole: vi.fn().mockReturnValue({ elementHandles }),
  } as unknown as Locator;
  const addLocatorHandler = vi
    .fn<Page["addLocatorHandler"]>()
    .mockResolvedValue(undefined);
  const page = {
    getByTestId: vi.fn().mockReturnValue(intro),
    addLocatorHandler,
  } as unknown as Page;

  await dismissTabIntros(page);
  const handler = addLocatorHandler.mock.calls[0][1];
  return { elementHandles, invoke: () => handler(intro) };
}

describe("dismissTabIntros", () => {
  it.each([
    "locator.elementHandles: Unable to adopt element handle from a different document",
    "locator.elementHandles: Execution context was destroyed, most likely because of a navigation",
  ])("recovers from a navigation during button lookup: %s", async (message) => {
    const { elementHandles, invoke } = await registerIntroHandler();
    let rejectLookup!: (error: Error) => void;
    const lookup = new Promise<ButtonHandle[]>((_, reject) => {
      rejectLookup = reject;
    });
    const newDocumentButton = createButton();
    elementHandles
      .mockReturnValueOnce(lookup)
      .mockResolvedValueOnce([newDocumentButton as unknown as ButtonHandle]);

    const interruptedHandler = invoke();
    const result = expect(interruptedHandler).resolves.toBeUndefined();
    rejectLookup(new Error(message));
    await result;

    await invoke();

    expect(elementHandles).toHaveBeenCalledTimes(2);
    expect(newDocumentButton.press).toHaveBeenCalledExactlyOnceWith("Enter");
    expect(newDocumentButton.dispose).toHaveBeenCalledOnce();
  });

  it.each([
    "Element is not attached to the DOM",
    "Execution context was destroyed, most likely because of a navigation",
  ])("releases a button detached during dismissal: %s", async (message) => {
    const { elementHandles, invoke } = await registerIntroHandler();
    const oldButton = createButton();
    const newButton = createButton();
    oldButton.press.mockRejectedValueOnce(new Error(message));
    elementHandles
      .mockResolvedValueOnce([oldButton as unknown as ButtonHandle])
      .mockResolvedValueOnce([newButton as unknown as ButtonHandle]);

    await expect(invoke()).resolves.toBeUndefined();
    expect(oldButton.dispose).toHaveBeenCalledOnce();
    await invoke();

    expect(newButton.press).toHaveBeenCalledExactlyOnceWith("Enter");
    expect(newButton.dispose).toHaveBeenCalledOnce();
  });

  it("allows an overlay to disappear before its button is found", async () => {
    const { elementHandles, invoke } = await registerIntroHandler();
    elementHandles.mockResolvedValueOnce([]);

    await expect(invoke()).resolves.toBeUndefined();
  });

  it.each([
    new Error("Target page, context or browser has been closed"),
    "unexpected lookup failure",
  ])("preserves unexpected lookup failures: %s", async (error) => {
    const { elementHandles, invoke } = await registerIntroHandler();
    elementHandles.mockRejectedValueOnce(error);

    await expect(invoke()).rejects.toBe(error);
  });

  it("preserves unexpected press failures and releases the handle", async () => {
    const { elementHandles, invoke } = await registerIntroHandler();
    const button = createButton();
    const error = new Error("Unexpected keyboard failure");
    elementHandles.mockResolvedValueOnce([button as unknown as ButtonHandle]);
    button.press.mockRejectedValueOnce(error);

    await expect(invoke()).rejects.toBe(error);
    expect(button.dispose).toHaveBeenCalledOnce();
  });
});
