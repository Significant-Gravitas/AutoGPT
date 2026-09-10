import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DeviceAuthCredentialsModal } from "../DeviceAuthCredentialsModal";

const connectProps = vi.hoisted(() => ({ current: null as any }));

vi.mock("@/components/contextual/DeviceAuth/DeviceAuthConnectButton", () => ({
  DeviceAuthConnectButton: (props: any) => {
    connectProps.current = props;
    return <div data-testid="device-auth-connect">{props.providerName}</div>;
  },
}));

afterEach(() => {
  cleanup();
  connectProps.current = null;
});

function renderModal(overrides: Record<string, unknown> = {}) {
  const props = {
    open: true,
    onClose: vi.fn(),
    provider: "stripe_link",
    providerName: "Stripe Link",
    onCredentialsCreate: vi.fn(),
    ...overrides,
  };
  render(<DeviceAuthCredentialsModal {...(props as any)} />);
  return props;
}

describe("DeviceAuthCredentialsModal", () => {
  it("hosts the device auth flow for the block's provider", () => {
    renderModal();

    expect(screen.getByTestId("device-auth-connect")).toBeTruthy();
    expect(connectProps.current.provider).toBe("stripe_link");
  });

  // The whole point of routing device_code here: the node has to end up wired
  // to the credential the user just approved on their phone.
  it("selects the credential the poll returned", () => {
    const props = renderModal();

    connectProps.current.onSuccess({
      id: "cred-99",
      type: "oauth2",
      provider: "stripe_link",
      title: "Stripe Link",
    });

    expect(props.onCredentialsCreate).toHaveBeenCalledWith({
      id: "cred-99",
      type: "oauth2",
      provider: "stripe_link",
      title: "Stripe Link",
    });
    expect(props.onClose).toHaveBeenCalled();
  });

  it("closes without selecting anything when no credential came back", () => {
    const props = renderModal();

    connectProps.current.onSuccess(undefined);

    expect(props.onCredentialsCreate).not.toHaveBeenCalled();
    expect(props.onClose).toHaveBeenCalled();
  });

  it("tolerates a credential with no title", () => {
    const props = renderModal();

    connectProps.current.onSuccess({
      id: "cred-1",
      type: "oauth2",
      provider: "stripe_link",
      title: null,
    });

    expect(props.onCredentialsCreate).toHaveBeenCalledWith(
      expect.objectContaining({ id: "cred-1", title: undefined }),
    );
  });
});
