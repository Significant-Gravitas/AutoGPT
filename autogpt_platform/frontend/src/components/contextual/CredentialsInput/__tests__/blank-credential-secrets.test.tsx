import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import useCredentials from "@/hooks/useCredentials";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaResponse,
} from "@/lib/autogpt-server-api/types";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";

import { APIKeyCredentialsModal } from "../components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { HostScopedCredentialsModal } from "../components/HotScopedCredentialsModal/HotScopedCredentialsModal";
import { PasswordCredentialsModal } from "../components/PasswordCredentialsModal/PasswordCredentialsModal";

// #15727: the builder's credential dialogs must not save blank secrets, and
// must send what the user typed without surrounding whitespace.

vi.mock("@/hooks/useCredentials", () => ({ default: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn(), toasts: [] }),
}));

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;
const mockToast = toast as unknown as ReturnType<typeof vi.fn>;

const schema = {
  type: "object",
  properties: {},
  credentials_provider: ["openai"],
  credentials_types: ["api_key"],
} as unknown as BlockIOCredentialsSubSchema;

const createAPIKeyCredentials = vi.fn();
const createUserPasswordCredentials = vi.fn();
const createHostScopedCredentials = vi.fn();
const deleteCredentials = vi.fn();

beforeEach(() => {
  createAPIKeyCredentials.mockResolvedValue({ id: "new", title: "Key" });
  createUserPasswordCredentials.mockResolvedValue({ id: "new", title: "U" });
  createHostScopedCredentials.mockResolvedValue({ id: "new", title: "H" });
  deleteCredentials.mockResolvedValue(undefined);
  mockUseCredentials.mockReturnValue({
    isLoading: false,
    provider: "openai",
    providerName: "OpenAI",
    supportsApiKey: true,
    supportsUserPassword: true,
    supportsHostScoped: true,
    discriminatorValue: undefined,
    savedCredentials: [],
    createAPIKeyCredentials,
    createUserPasswordCredentials,
    createHostScopedCredentials,
    deleteCredentials,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// The API key and host fields show no message when refused, but once
// validation fails react-hook-form moves the cursor into the refused field.
// The button is focused first, so only the refusal can move the cursor there.
async function submitAndWaitForRefusal(
  button: HTMLElement,
  refusedField: HTMLElement,
) {
  button.focus();
  fireEvent.click(button);
  await waitFor(() => expect(document.activeElement).toBe(refusedField));
}

function type(placeholder: string, value: string) {
  fireEvent.change(screen.getByPlaceholderText(placeholder), {
    target: { value },
  });
}

describe("APIKeyCredentialsModal", () => {
  function renderModal() {
    const onCredentialsCreate = vi.fn();
    render(
      <APIKeyCredentialsModal
        schema={schema}
        open
        onClose={vi.fn()}
        onCredentialsCreate={onCredentialsCreate}
      />,
    );
    return onCredentialsCreate;
  }

  it("rejects a spaces-only API key without saving", async () => {
    renderModal();
    type("Enter a name for this API Key...", "My key");
    type("Enter API Key...", "    ");

    await submitAndWaitForRefusal(
      screen.getByRole("button", { name: "Add API Key" }),
      screen.getByPlaceholderText("Enter API Key..."),
    );
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  it("rejects a spaces-only name without saving", async () => {
    renderModal();
    type("Enter a name for this API Key...", "   ");
    type("Enter API Key...", "sk-abc");

    await submitAndWaitForRefusal(
      screen.getByRole("button", { name: "Add API Key" }),
      screen.getByPlaceholderText("Enter a name for this API Key..."),
    );
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  it("sends the key and name trimmed", async () => {
    const onCredentialsCreate = renderModal();
    type("Enter a name for this API Key...", "  My key ");
    type("Enter API Key...", "  sk-abc  ");

    fireEvent.click(screen.getByRole("button", { name: "Add API Key" }));

    await waitFor(() => expect(createAPIKeyCredentials).toHaveBeenCalled());
    expect(createAPIKeyCredentials).toHaveBeenCalledWith({
      api_key: "sk-abc",
      title: "My key",
      expires_at: undefined,
    });
    await waitFor(() => expect(onCredentialsCreate).toHaveBeenCalled());
  });
});

describe("PasswordCredentialsModal", () => {
  function renderModal() {
    render(
      <PasswordCredentialsModal
        schema={schema}
        open
        onClose={vi.fn()}
        onCredentialsCreate={vi.fn()}
      />,
    );
  }

  it("rejects a spaces-only username without saving", async () => {
    renderModal();
    type("Enter username...", "   ");
    type("Enter password...", "pw");
    type("Enter a name for this user login...", "Login");

    fireEvent.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );

    expect(await screen.findByText("Username is required")).toBeDefined();
    expect(createUserPasswordCredentials).not.toHaveBeenCalled();
  });

  it("rejects a spaces-only name without saving", async () => {
    renderModal();
    type("Enter username...", "alice");
    type("Enter password...", "pw");
    type("Enter a name for this user login...", "   ");

    fireEvent.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );

    expect(await screen.findByText("Name is required")).toBeDefined();
    expect(createUserPasswordCredentials).not.toHaveBeenCalled();
  });

  it("trims the username but sends the password as typed", async () => {
    renderModal();
    type("Enter username...", " alice ");
    type("Enter password...", " pass word ");
    type("Enter a name for this user login...", "Login");

    fireEvent.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );

    await waitFor(() =>
      expect(createUserPasswordCredentials).toHaveBeenCalled(),
    );
    expect(createUserPasswordCredentials.mock.calls[0][0]).toMatchObject({
      username: "alice",
      password: " pass word ",
    });
  });
});

describe("HostScopedCredentialsModal", () => {
  const existing: CredentialsMetaResponse = {
    id: "existing-cred",
    provider: "openai",
    type: "host_scoped",
    title: "api.example.com",
    host: "api.example.com",
  };

  const providers: CredentialsProvidersContextType = {
    openai: {
      provider: "openai",
      providerName: "OpenAI",
      savedCredentials: [existing],
      isSystemProvider: false,
      oAuthCallback: vi.fn(),
      mcpOAuthCallback: vi.fn(),
      mcpStoreToken: vi.fn(),
      createAPIKeyCredentials,
      createUserPasswordCredentials,
      createHostScopedCredentials,
      deleteCredentials,
    },
  };

  function renderModal() {
    render(
      <CredentialsProvidersContext.Provider value={providers}>
        <HostScopedCredentialsModal
          schema={schema}
          open
          onClose={vi.fn()}
          onCredentialsCreate={vi.fn()}
        />
      </CredentialsProvidersContext.Provider>,
    );
  }

  function submitButton() {
    return screen.getByRole("button", {
      name: /^(Save|Update) & use these credentials$/,
    }) as HTMLButtonElement;
  }

  // The backend strips with Python's str.strip(), which also removes U+0085
  // and U+001C-U+001F, so it refuses a header made only of them as blank.
  it.each([
    ["a value of spaces", "Authorization", "   "],
    ["a value of U+0085", "Authorization", "\u0085"],
    ["a name of U+001F", "\u001f", "Bearer x"],
  ])(
    "does not delete the existing credential when the only header has %s",
    async (_, name, value) => {
      renderModal();
      type("Enter host (e.g., api.example.com)", "api.example.com");
      type("Header name (e.g., Authorization)", name);
      type("Header value (e.g., Bearer token123)", value);

      expect(submitButton().disabled).toBe(true);
      fireEvent.submit(submitButton().closest("form")!);

      await waitFor(() =>
        expect(mockToast).toHaveBeenCalledWith(
          expect.objectContaining({ title: "Add at least one header" }),
        ),
      );
      expect(deleteCredentials).not.toHaveBeenCalled();
      expect(createHostScopedCredentials).not.toHaveBeenCalled();
    },
  );

  it.each([
    ["exactly", "Authorization"],
    ["with padding", " Authorization "],
    ["with a trailing U+0085", "Authorization\u0085"],
    ["in another case", "authorization"],
  ])(
    "does not delete the existing credential when a header name repeats %s",
    async (_, repeatedName) => {
      renderModal();
      type("Enter host (e.g., api.example.com)", "api.example.com");
      fireEvent.click(
        screen.getByRole("button", { name: "Add Another Header" }),
      );
      const names = screen.getAllByPlaceholderText(
        "Header name (e.g., Authorization)",
      );
      const values = screen.getAllByPlaceholderText(
        "Header value (e.g., Bearer token123)",
      );
      fireEvent.change(names[0], { target: { value: "Authorization" } });
      fireEvent.change(values[0], { target: { value: "Bearer x" } });
      fireEvent.change(names[1], { target: { value: repeatedName } });
      fireEvent.change(values[1], { target: { value: "Bearer y" } });

      fireEvent.click(submitButton());

      await waitFor(() =>
        expect(mockToast).toHaveBeenCalledWith(
          expect.objectContaining({ title: "Use each header name once" }),
        ),
      );
      expect(deleteCredentials).not.toHaveBeenCalled();
      expect(createHostScopedCredentials).not.toHaveBeenCalled();
    },
  );

  it("trims the host before replacing the existing credential", async () => {
    renderModal();
    type("Enter host (e.g., api.example.com)", "  api.example.com  ");
    type("Header name (e.g., Authorization)", "Authorization");
    type("Header value (e.g., Bearer token123)", " Bearer x ");

    expect(submitButton().disabled).toBe(false);
    fireEvent.click(submitButton());

    await waitFor(() => expect(createHostScopedCredentials).toHaveBeenCalled());
    expect(deleteCredentials).toHaveBeenCalledWith("existing-cred", true);
    expect(createHostScopedCredentials).toHaveBeenCalledWith({
      host: "api.example.com",
      title: "api.example.com",
      headers: { Authorization: "Bearer x" },
    });
  });

  it("rejects a spaces-only host", async () => {
    renderModal();
    type("Enter host (e.g., api.example.com)", "   ");
    type("Header name (e.g., Authorization)", "Authorization");
    type("Header value (e.g., Bearer token123)", "Bearer x");

    await submitAndWaitForRefusal(
      submitButton(),
      screen.getByPlaceholderText("Enter host (e.g., api.example.com)"),
    );
    expect(deleteCredentials).not.toHaveBeenCalled();
    expect(createHostScopedCredentials).not.toHaveBeenCalled();
  });
});
