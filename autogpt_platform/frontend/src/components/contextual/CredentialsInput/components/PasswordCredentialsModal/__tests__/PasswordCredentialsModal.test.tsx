import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";

import { PasswordCredentialsModal } from "../PasswordCredentialsModal";

vi.mock("@/hooks/useCredentials", () => ({ default: vi.fn() }));

import useCredentials from "@/hooks/useCredentials";

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderModal(provider: string, providerName: string) {
  const createUserPasswordCredentials = vi.fn().mockResolvedValue({
    id: "cred-1",
    title: "Mine",
  });
  mockUseCredentials.mockReturnValue({
    provider,
    providerName,
    isLoading: false,
    supportsUserPassword: true,
    createUserPasswordCredentials,
  });
  const onCredentialsCreate = vi.fn();
  const schema = {
    type: "object",
    properties: {},
    credentials_provider: [provider],
    credentials_types: ["user_password"],
  } as unknown as BlockIOCredentialsSubSchema;
  render(
    <PasswordCredentialsModal
      schema={schema}
      open
      onClose={vi.fn()}
      onCredentialsCreate={onCredentialsCreate}
    />,
  );
  return { createUserPasswordCredentials, onCredentialsCreate };
}

async function fillLogin(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Username"), "postmaster");
  await user.type(
    screen.getByLabelText("Password", { selector: "input" }),
    "<test-password>",
  );
  await user.type(screen.getByLabelText("Name"), "Mine");
}

const SMTP_SERVER_MESSAGE =
  "Enter a valid server, like smtp.example.com or smtp.example.com:587 (port 1-65535)";
const DATABASE_SERVER_MESSAGE =
  "Enter a valid server, like db.example.com or db.example.com:5432 (port 1-65535)";

async function saveAndExpectRefused(
  user: ReturnType<typeof userEvent.setup>,
  createUserPasswordCredentials: ReturnType<typeof vi.fn>,
  message: string,
) {
  await user.click(
    screen.getByRole("button", { name: "Save & use this user login" }),
  );
  expect(await screen.findByText(message)).toBeDefined();
  expect(screen.getByLabelText("Server")).toBeDefined();
  expect(createUserPasswordCredentials).not.toHaveBeenCalled();
}

describe("PasswordCredentialsModal for a login sent to a server the workflow names", () => {
  it("requires the server for an SMTP login", async () => {
    const { createUserPasswordCredentials } = renderModal("smtp", "SMTP");
    const user = userEvent.setup();

    expect(await screen.findByLabelText("Server")).toBeDefined();
    await fillLogin(user);
    await saveAndExpectRefused(
      user,
      createUserPasswordCredentials,
      "Server is required",
    );

    await user.type(screen.getByLabelText("Server"), "smtp.example.com");
    await user.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );
    await waitFor(() =>
      expect(createUserPasswordCredentials).toHaveBeenCalledTimes(1),
    );
    expect(createUserPasswordCredentials).toHaveBeenCalledWith(
      expect.objectContaining({ scope: "smtp://smtp.example.com" }),
    );
  });

  it("saves an SMTP login scoped to smtp://<server>", async () => {
    const { createUserPasswordCredentials, onCredentialsCreate } = renderModal(
      "smtp",
      "SMTP",
    );
    const user = userEvent.setup();

    await user.type(
      await screen.findByLabelText("Server"),
      "SMTP.Example.com:587",
    );
    await fillLogin(user);
    await user.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );

    await waitFor(() =>
      expect(createUserPasswordCredentials).toHaveBeenCalledWith({
        username: "postmaster",
        password: "<test-password>",
        title: "Mine",
        scope: "smtp://smtp.example.com:587",
      }),
    );
    await waitFor(() =>
      expect(onCredentialsCreate).toHaveBeenCalledWith({
        provider: "smtp",
        id: "cred-1",
        type: "user_password",
        title: "Mine",
      }),
    );
  });

  it("requires the server for a database login", async () => {
    const { createUserPasswordCredentials } = renderModal(
      "database",
      "Database",
    );
    const user = userEvent.setup();

    expect(await screen.findByLabelText("Server")).toBeDefined();
    await fillLogin(user);
    await saveAndExpectRefused(
      user,
      createUserPasswordCredentials,
      "Server is required",
    );

    await user.type(screen.getByLabelText("Server"), "db.example.com");
    await user.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );
    await waitFor(() =>
      expect(createUserPasswordCredentials).toHaveBeenCalledTimes(1),
    );
    expect(createUserPasswordCredentials).toHaveBeenCalledWith(
      expect.objectContaining({ scope: "postgresql://db.example.com" }),
    );
  });

  it("saves a database login scoped to the engine's scheme", async () => {
    const { createUserPasswordCredentials } = renderModal(
      "database",
      "Database",
    );
    const user = userEvent.setup();

    expect(screen.getByText("Database type")).toBeDefined();
    await user.type(
      await screen.findByLabelText("Server"),
      "db.example.com:5432",
    );
    await fillLogin(user);
    await user.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );

    await waitFor(() =>
      expect(createUserPasswordCredentials).toHaveBeenCalledWith(
        expect.objectContaining({
          scope: "postgresql://db.example.com:5432",
        }),
      ),
    );
  });

  it.each([
    ["smtp", "SMTP", "smtp://smtp.qa-own.invalid", SMTP_SERVER_MESSAGE],
    ["smtp", "SMTP", "smtp.example.com:99999", SMTP_SERVER_MESSAGE],
    ["database", "Database", "db.example.com:0", DATABASE_SERVER_MESSAGE],
  ])(
    "shows why a %s server %s is refused and keeps the modal open",
    async (provider, providerName, server, message) => {
      const { createUserPasswordCredentials } = renderModal(
        provider,
        providerName,
      );
      const user = userEvent.setup();

      await user.type(await screen.findByLabelText("Server"), server);
      await fillLogin(user);
      await saveAndExpectRefused(user, createUserPasswordCredentials, message);
      expect(
        screen.getByText(`Add new username & password for ${providerName}`),
      ).toBeDefined();

      const field = screen.getByLabelText("Server");
      await user.clear(field);
      await user.type(field, "host.example.com:587");
      await waitFor(() => expect(screen.queryByText(message)).toBeNull());
      await user.click(
        screen.getByRole("button", { name: "Save & use this user login" }),
      );
      await waitFor(() =>
        expect(createUserPasswordCredentials).toHaveBeenCalledTimes(1),
      );
    },
  );
});

describe("PasswordCredentialsModal for a provider with a fixed server", () => {
  it("asks for no server and sends no scope", async () => {
    const { createUserPasswordCredentials } = renderModal(
      "webshare_proxy",
      "Webshare Proxy",
    );
    const user = userEvent.setup();

    await screen.findByLabelText("Username");
    expect(screen.queryByLabelText("Server")).toBeNull();
    await fillLogin(user);
    await user.click(
      screen.getByRole("button", { name: "Save & use this user login" }),
    );

    await waitFor(() =>
      expect(createUserPasswordCredentials).toHaveBeenCalledWith({
        username: "postmaster",
        password: "<test-password>",
        title: "Mine",
      }),
    );
  });
});
