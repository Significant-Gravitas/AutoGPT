import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConnectCredentialDialog } from "../ConnectCredentialDialog";

vi.mock(
  "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect",
  () => ({ useOAuthConnect: vi.fn() }),
);
vi.mock(
  "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useApiKeyConnectForm",
  () => ({ useApiKeyConnectForm: vi.fn() }),
);

vi.mock(
  "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/ConnectMethodView/ConnectMethodView",
  () => ({
    ConnectMethodView: ({
      provider,
      selectedMethod,
      onSelectMethod,
      onDeviceAuthSuccess,
    }: {
      provider: { id: string; name: string; supportedAuthTypes: string[] };
      selectedMethod: string | null;
      onSelectMethod: (method: string) => void;
      onDeviceAuthSuccess: () => void;
    }) => (
      <div data-testid="connect-method-view">
        <span>Connect AutoGPT to {provider.name}</span>
        <span data-testid="auth-methods">
          {provider.supportedAuthTypes.join(",")}
        </span>
        <span data-testid="selected-method">{selectedMethod ?? "none"}</span>
        {provider.supportedAuthTypes.map((method) => (
          <button key={method} onClick={() => onSelectMethod(method)}>
            {`select-${method}`}
          </button>
        ))}
        {provider.supportedAuthTypes.includes("device_code") && (
          <button onClick={onDeviceAuthSuccess}>complete-device_code</button>
        )}
      </div>
    ),
  }),
);

vi.mock("@/components/molecules/Dialog/Dialog", () => {
  function MockDialog({
    children,
    controlled,
  }: {
    children: React.ReactNode;
    controlled?: { isOpen: boolean; set: (open: boolean) => void };
  }) {
    if (!controlled?.isOpen) return null;
    return (
      <div role="dialog">
        <button
          data-testid="dialog-dismiss"
          onClick={() => controlled.set(false)}
        >
          dismiss
        </button>
        {children}
      </div>
    );
  }
  MockDialog.Content = function Content({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return <div>{children}</div>;
  };
  return { Dialog: MockDialog };
});

import { useApiKeyConnectForm } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useApiKeyConnectForm";
import { useOAuthConnect } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/useOAuthConnect";

const mockUseOAuthConnect = useOAuthConnect as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseApiKeyConnectForm = useApiKeyConnectForm as unknown as ReturnType<
  typeof vi.fn
>;

function makeOAuthReturn(overrides: Partial<{ isPending: boolean }> = {}) {
  return {
    connect: vi.fn(),
    isPending: overrides.isPending ?? false,
  };
}

function makeApiKeyReturn(
  overrides: Partial<{ isValid: boolean; isPending: boolean }> = {},
) {
  return {
    form: {
      reset: vi.fn(),
      formState: { isValid: overrides.isValid ?? false },
      handleSubmit: (onValid: (values: unknown) => void) => () =>
        onValid({ title: "Key", apiKey: "sk-123", expiresAt: "" }),
    },
    handleSubmit: vi.fn(),
    isPending: overrides.isPending ?? false,
  };
}

const baseSchema = {
  credentials_provider: ["github"],
  credentials_types: ["api_key", "oauth2"],
} as BlockIOCredentialsSubSchema;

function renderDialog(
  overrides: Partial<React.ComponentProps<typeof ConnectCredentialDialog>> = {},
) {
  const onClose = vi.fn();
  const utils = render(
    <ConnectCredentialDialog
      schema={baseSchema}
      provider="github"
      displayName="GitHub"
      open
      onClose={onClose}
      {...overrides}
    />,
  );
  return { onClose, ...utils };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseOAuthConnect.mockReturnValue(makeOAuthReturn());
  mockUseApiKeyConnectForm.mockReturnValue(makeApiKeyReturn());
});

afterEach(cleanup);

describe("ConnectCredentialDialog", () => {
  it("renders nothing while closed", () => {
    renderDialog({ open: false });
    expect(screen.queryByTestId("connect-method-view")).toBeNull();
  });

  it("renders the method view scoped to the provider's known auth methods", () => {
    renderDialog({
      schema: {
        credentials_provider: ["github"],
        credentials_types: ["api_key", "oauth2", "not_a_method"],
      } as unknown as BlockIOCredentialsSubSchema,
    });

    expect(screen.getByText("Connect AutoGPT to GitHub")).toBeDefined();
    expect(screen.getByTestId("auth-methods").textContent).toBe(
      "api_key,oauth2",
    );
  });

  it("offers device auth instead of the stored OAuth credential shape", () => {
    const apiKey = makeApiKeyReturn();
    mockUseApiKeyConnectForm.mockReturnValue(apiKey);
    const { onClose } = renderDialog({
      schema: {
        credentials_provider: ["stripe_link"],
        credentials_types: ["oauth2", "device_code"],
      } as BlockIOCredentialsSubSchema,
    });

    expect(screen.getByTestId("auth-methods").textContent).toBe("device_code");
    expect(screen.queryByText("select-oauth2")).toBeNull();

    fireEvent.click(screen.getByText("complete-device_code"));

    expect(onClose).toHaveBeenCalledOnce();
    expect(apiKey.form.reset).toHaveBeenCalledOnce();
  });

  it("disables Continue until a method is selected", () => {
    renderDialog();

    const cont = screen.getByText("Continue").closest("button");
    expect(cont?.disabled).toBe(true);

    fireEvent.click(screen.getByText("select-oauth2"));
    expect(screen.getByTestId("selected-method").textContent).toBe("oauth2");
    expect(screen.getByText("Continue").closest("button")?.disabled).toBe(
      false,
    );
  });

  it("starts the OAuth flow when Continue is clicked with OAuth selected", () => {
    const oauth = makeOAuthReturn();
    mockUseOAuthConnect.mockReturnValue(oauth);
    renderDialog();

    fireEvent.click(screen.getByText("select-oauth2"));
    fireEvent.click(screen.getByText("Continue"));

    expect(oauth.connect).toHaveBeenCalledOnce();
  });

  it("keeps Continue disabled for an invalid API key form", () => {
    mockUseApiKeyConnectForm.mockReturnValue(
      makeApiKeyReturn({ isValid: false }),
    );
    renderDialog();

    fireEvent.click(screen.getByText("select-api_key"));
    expect(screen.getByText("Continue").closest("button")?.disabled).toBe(true);
  });

  it("submits the API key form from Continue once valid", () => {
    const apiKey = makeApiKeyReturn({ isValid: true });
    mockUseApiKeyConnectForm.mockReturnValue(apiKey);
    renderDialog();

    fireEvent.click(screen.getByText("select-api_key"));
    fireEvent.click(screen.getByText("Continue"));

    expect(apiKey.handleSubmit).toHaveBeenCalledWith({
      title: "Key",
      apiKey: "sk-123",
      expiresAt: "",
    });
  });

  it("hides Continue for methods it cannot drive", () => {
    renderDialog({
      schema: {
        credentials_provider: ["github"],
        credentials_types: ["user_password"],
      } as BlockIOCredentialsSubSchema,
    });

    fireEvent.click(screen.getByText("select-user_password"));
    expect(screen.queryByText("Continue")).toBeNull();
  });

  it("shows the connecting state while OAuth is pending", () => {
    mockUseOAuthConnect.mockReturnValue(makeOAuthReturn({ isPending: true }));
    renderDialog();

    expect(screen.getByText("Connecting…")).toBeDefined();
  });

  it("resets the picked method and form on Cancel", () => {
    const apiKey = makeApiKeyReturn();
    mockUseApiKeyConnectForm.mockReturnValue(apiKey);
    const { onClose } = renderDialog();

    fireEvent.click(screen.getByText("select-oauth2"));
    fireEvent.click(screen.getByText("Cancel"));

    expect(onClose).toHaveBeenCalledOnce();
    expect(apiKey.form.reset).toHaveBeenCalledOnce();
  });

  it("resets and closes when the dialog itself is dismissed", () => {
    const apiKey = makeApiKeyReturn();
    mockUseApiKeyConnectForm.mockReturnValue(apiKey);
    const { onClose } = renderDialog();

    fireEvent.click(screen.getByTestId("dialog-dismiss"));

    expect(onClose).toHaveBeenCalledOnce();
    expect(apiKey.form.reset).toHaveBeenCalledOnce();
  });

  it("resets and closes once a connection succeeds", () => {
    const apiKey = makeApiKeyReturn();
    mockUseApiKeyConnectForm.mockReturnValue(apiKey);
    const { onClose } = renderDialog();

    const { onSuccess } = mockUseOAuthConnect.mock.calls[0][0] as {
      onSuccess: () => void;
    };
    act(() => onSuccess());

    expect(onClose).toHaveBeenCalledOnce();
    expect(apiKey.form.reset).toHaveBeenCalledOnce();
  });
});
