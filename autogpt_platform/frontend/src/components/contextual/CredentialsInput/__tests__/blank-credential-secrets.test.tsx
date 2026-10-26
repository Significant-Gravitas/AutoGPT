import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import useCredentials from "@/hooks/useCredentials";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import { toast } from "@/components/molecules/Toast/use-toast";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";

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

// The dialogs' inputs don't render zod messages here, so let react-hook-form
// finish validating and then check that nothing was sent.
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
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

    fireEvent.click(screen.getByRole("button", { name: "Add API Key" }));

    await settle();
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  it("rejects a spaces-only name without saving", async () => {
    renderModal();
    type("Enter a name for this API Key...", "   ");
    type("Enter API Key...", "sk-abc");

    fireEvent.click(screen.getByRole("button", { name: "Add API Key" }));

    await settle();
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  // #16234: an expiry in the past is refused before anything is saved.
  it("rejects a past expiry without saving", async () => {
    renderModal();
    type("Enter a name for this API Key...", "My key");
    type("Enter API Key...", "sk-abc");
    type("Select expiration date...", "2020-01-01T10:00");

    fireEvent.click(screen.getByRole("button", { name: "Add API Key" }));

    expect(
      await screen.findByText("Expiry must be in the future"),
    ).toBeDefined();
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  it("saves a future expiry as local midnight of that day", async () => {
    renderModal();
    type("Enter a name for this API Key...", "My key");
    type("Enter API Key...", "sk-abc");
    type("Select expiration date...", "2999-01-01T10:00");

    fireEvent.click(screen.getByRole("button", { name: "Add API Key" }));

    await waitFor(() =>
      expect(createAPIKeyCredentials).toHaveBeenCalledWith(
        expect.objectContaining({
          expires_at: new Date(2999, 0, 1).getTime() / 1000,
        }),
      ),
    );
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

    await settle();
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
  const existing = {
    id: "existing-cred",
    provider: "openai",
    type: "host_scoped",
    title: "api.example.com",
    host: "api.example.com",
  };

  function renderModal() {
    render(
      <CredentialsProvidersContext.Provider
        value={
          {
            openai: { savedCredentials: [existing] },
          } as any
        }
      >
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
      name: /use these credentials/,
    }) as HTMLButtonElement;
  }

  it("does not delete the existing credential when every header is blank", async () => {
    renderModal();
    type("Enter host (e.g., api.example.com)", "api.example.com");
    type("Header name (e.g., Authorization)", "Authorization");
    type("Header value (e.g., Bearer token123)", "   ");

    expect(submitButton().disabled).toBe(true);
    fireEvent.submit(submitButton().closest("form")!);

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Add at least one header" }),
      ),
    );
    expect(deleteCredentials).not.toHaveBeenCalled();
    expect(createHostScopedCredentials).not.toHaveBeenCalled();
  });

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

    fireEvent.click(submitButton());

    await settle();
    expect(deleteCredentials).not.toHaveBeenCalled();
    expect(createHostScopedCredentials).not.toHaveBeenCalled();
  });
});
