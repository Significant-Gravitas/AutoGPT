import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV1ListUserApiKeysMockHandler,
  getPostV1CreateNewApiKeyMockHandler200,
  getPostV1CreateNewApiKeyMockHandler422,
  getPostV1CreateNewApiKeyResponseMock200,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";

import SettingsApiKeysPage from "../page";

const toastSpy = vi.fn();

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

function openCreateDialog() {
  const createButtons = screen.getAllByRole("button", {
    name: /^create key$/i,
  });
  fireEvent.click(createButtons[0]);
}

describe("SettingsApiKeysPage - create flow", () => {
  beforeEach(() => {
    toastSpy.mockClear();
  });

  test("opens the create dialog with a form when Create Key is clicked", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/create api key/i)).toBeDefined();
    expect(within(dialog).getByLabelText(/^name$/i)).toBeDefined();
    expect(within(dialog).getByLabelText(/description/i)).toBeDefined();
    expect(within(dialog).getByText(/permissions/i)).toBeDefined();
  });

  test("toggling a permission off re-disables the submit button", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "Toggle Key" },
    });
    const checkbox = within(dialog).getByRole("checkbox", {
      name: /execute graph/i,
    });

    fireEvent.click(checkbox);
    const submit = within(dialog).getByRole("button", {
      name: /create key/i,
    }) as HTMLButtonElement;
    await waitFor(() => {
      expect(submit.disabled).toBe(false);
    });

    fireEvent.click(checkbox);
    await waitFor(() => {
      expect(submit.disabled).toBe(true);
    });
  });

  test("disables submit until the form is valid", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    const submit = within(dialog).getByRole("button", { name: /create key/i });
    expect((submit as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "Integration Test Key" },
    });
    // Still invalid — no permission picked yet.
    await waitFor(() => {
      expect((submit as HTMLButtonElement).disabled).toBe(true);
    });

    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );

    await waitFor(() => {
      expect((submit as HTMLButtonElement).disabled).toBe(false);
    });
  });

  test("submits successfully and switches to the success view with plain text key", async () => {
    const plain = "plain-secret-key-abc123";
    server.use(
      getGetV1ListUserApiKeysMockHandler([]),
      getPostV1CreateNewApiKeyMockHandler200(
        getPostV1CreateNewApiKeyResponseMock200({ plain_text_key: plain }),
      ),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "My New Key" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );

    const submit = within(dialog).getByRole("button", {
      name: /create key/i,
    }) as HTMLButtonElement;
    await waitFor(() => {
      expect(submit.disabled).toBe(false);
    });
    fireEvent.click(submit);

    expect(await screen.findByText(/your new api key/i)).toBeDefined();
    expect(await screen.findByText(plain)).toBeDefined();
    expect(
      within(dialog).getAllByRole("button", { name: /^close$/i }).length,
    ).toBeGreaterThan(0);
  });

  test("dismisses the dialog via the header Close button when not submitting", async () => {
    server.use(getGetV1ListUserApiKeysMockHandler([]));

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    fireEvent.click(within(dialog).getByRole("button", { name: /^close$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });

  test("keeps the form open when the API returns 422", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([]),
      getPostV1CreateNewApiKeyMockHandler422(),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    const dialog = await screen.findByRole("dialog");

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "Should Fail" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );

    const submit = within(dialog).getByRole("button", {
      name: /create key/i,
    }) as HTMLButtonElement;
    await waitFor(() => {
      expect(submit.disabled).toBe(false);
    });
    fireEvent.click(submit);

    // No transition to the success view — form inputs remain mounted.
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/^name$/i)).toBeDefined();
    });
    expect(screen.queryByText(/your new api key/i)).toBeNull();
  });

  test("reopening the dialog after a successful create resets the form", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([]),
      getPostV1CreateNewApiKeyMockHandler200(
        getPostV1CreateNewApiKeyResponseMock200({
          plain_text_key: "reset-me",
        }),
      ),
    );

    render(<SettingsApiKeysPage />);
    await screen.findByText(/no api key found/i);

    openCreateDialog();
    let dialog = await screen.findByRole("dialog");

    fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
      target: { value: "First Key" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /execute graph/i }),
    );

    const submit = within(dialog).getByRole("button", {
      name: /create key/i,
    }) as HTMLButtonElement;
    await waitFor(() => {
      expect(submit.disabled).toBe(false);
    });
    fireEvent.click(submit);

    await screen.findByText("reset-me");
    const closeButtons = within(dialog).getAllByRole("button", {
      name: /^close$/i,
    });
    fireEvent.click(closeButtons[closeButtons.length - 1]);

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });

    openCreateDialog();
    dialog = await screen.findByRole("dialog");

    const nameInput = within(dialog).getByLabelText(
      /^name$/i,
    ) as HTMLInputElement;
    expect(nameInput.value).toBe("");
    expect(within(dialog).queryByText(/your new api key/i)).toBeNull();
  });

  describe("success view", () => {
    const plain = "sk-super-secret-1234567890";
    const originalClipboard = Object.getOwnPropertyDescriptor(
      globalThis.navigator,
      "clipboard",
    );
    let writeTextSpy: ReturnType<typeof vi.fn>;

    async function submitUntilSuccessView() {
      server.use(
        getGetV1ListUserApiKeysMockHandler([]),
        getPostV1CreateNewApiKeyMockHandler200(
          getPostV1CreateNewApiKeyResponseMock200({ plain_text_key: plain }),
        ),
      );
      render(<SettingsApiKeysPage />);
      await screen.findByText(/no api key found/i);

      openCreateDialog();
      const dialog = await screen.findByRole("dialog");
      fireEvent.change(within(dialog).getByLabelText(/^name$/i), {
        target: { value: "Copyable Key" },
      });
      fireEvent.click(
        within(dialog).getByRole("checkbox", { name: /execute graph/i }),
      );
      const submit = within(dialog).getByRole("button", {
        name: /create key/i,
      }) as HTMLButtonElement;
      await waitFor(() => {
        expect(submit.disabled).toBe(false);
      });
      fireEvent.click(submit);
      await screen.findByText(plain);
      return dialog;
    }

    function installClipboard(writeText: ReturnType<typeof vi.fn>) {
      Object.defineProperty(globalThis.navigator, "clipboard", {
        configurable: true,
        value: { writeText },
      });
    }

    afterEach(() => {
      if (originalClipboard) {
        Object.defineProperty(
          globalThis.navigator,
          "clipboard",
          originalClipboard,
        );
      } else {
        // @ts-expect-error — ensure a missing clipboard stays missing across tests
        delete globalThis.navigator.clipboard;
      }
    });

    test("copies the plaintext key and shows a success toast when Copy is clicked", async () => {
      writeTextSpy = vi.fn().mockResolvedValue(undefined);
      installClipboard(writeTextSpy);

      const dialog = await submitUntilSuccessView();

      fireEvent.click(within(dialog).getByRole("button", { name: /^copy$/i }));

      await waitFor(() => {
        expect(writeTextSpy).toHaveBeenCalledWith(plain);
      });
      await waitFor(() => {
        expect(toastSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            title: "Copied to clipboard",
            variant: "success",
          }),
        );
      });
    });

    test("shows a destructive toast when the clipboard write fails", async () => {
      writeTextSpy = vi.fn().mockRejectedValue(new Error("denied"));
      installClipboard(writeTextSpy);

      const dialog = await submitUntilSuccessView();

      fireEvent.click(within(dialog).getByRole("button", { name: /^copy$/i }));

      await waitFor(() => {
        expect(toastSpy).toHaveBeenCalledWith(
          expect.objectContaining({
            title: "Could not copy to clipboard",
            variant: "destructive",
          }),
        );
      });
    });
  });
});
