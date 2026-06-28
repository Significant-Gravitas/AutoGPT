import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

vi.mock("@/hooks/useCredentials", () => ({ default: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: vi.fn(),
}));
vi.mock("@/services/scripts/scripts", () => ({
  loadScript: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  getGetV1GetSpecificCredentialByIdQueryOptions: vi.fn(),
  postV1GetPickerToken: vi.fn(),
}));
vi.mock("@/app/api/helpers", () => ({
  okData: vi.fn((resp: { status?: number; data?: unknown } | undefined) =>
    resp?.status === 200 ? resp.data : undefined,
  ),
}));
vi.mock("@/components/contextual/CredentialsInput/CredentialsInput", () => ({
  CredentialsInput: () => <div data-testid="credentials-input" />,
}));

import useCredentials from "@/hooks/useCredentials";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  getGetV1GetSpecificCredentialByIdQueryOptions,
  postV1GetPickerToken,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/lib/autogpt-server-api";
import { GoogleDrivePicker } from "../GoogleDrivePicker";

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseToast = useToast as unknown as ReturnType<typeof vi.fn>;
const mockPostPickerToken = postV1GetPickerToken as unknown as ReturnType<
  typeof vi.fn
>;
const mockGetQueryOptions =
  getGetV1GetSpecificCredentialByIdQueryOptions as unknown as ReturnType<
    typeof vi.fn
  >;

function makeCred(
  partial: Partial<CredentialsMetaResponse>,
): CredentialsMetaResponse {
  return {
    id: "cred-id",
    provider: "google",
    type: "oauth2",
    title: "Test",
    scopes: [],
    ...partial,
  } as CredentialsMetaResponse;
}

function setupGoogleGlobals() {
  const setOAuthToken = vi.fn();
  const setDeveloperKey = vi.fn();
  const setAppId = vi.fn();
  const setCallback = vi.fn();
  const enableFeature = vi.fn();
  const addView = vi.fn();
  const build = vi.fn().mockReturnValue({ setVisible: vi.fn() });

  class MockPickerBuilder {
    setOAuthToken = (...args: unknown[]) => {
      setOAuthToken(...args);
      return this;
    };
    setDeveloperKey = (...args: unknown[]) => {
      setDeveloperKey(...args);
      return this;
    };
    setAppId = (...args: unknown[]) => {
      setAppId(...args);
      return this;
    };
    setCallback = (...args: unknown[]) => {
      setCallback(...args);
      return this;
    };
    enableFeature = (...args: unknown[]) => {
      enableFeature(...args);
      return this;
    };
    addView = (...args: unknown[]) => {
      addView(...args);
      return this;
    };
    build = (...args: unknown[]) => build(...args);
  }

  class MockDocsView {
    setMode = vi.fn();
    constructor(_viewId?: unknown) {}
  }

  const mockedWindow = window as unknown as {
    gapi?: unknown;
    google?: unknown;
  };
  mockedWindow.gapi = {
    load: (_name: string, opts: { callback: () => void }) => opts.callback(),
  };
  mockedWindow.google = {
    accounts: {
      oauth2: {
        initTokenClient: vi.fn().mockReturnValue({
          requestAccessToken: vi.fn(),
        }),
      },
    },
    picker: {
      PickerBuilder: MockPickerBuilder,
      DocsView: MockDocsView,
      DocsViewMode: { LIST: "LIST" },
      Feature: { NAV_HIDDEN: "NAV_HIDDEN", MULTISELECT_ENABLED: "MULTI" },
      ViewId: {
        DOCS: "DOCS",
        DOCUMENTS: "DOCUMENTS",
        SPREADSHEETS: "SPREADSHEETS",
      },
      Response: { ACTION: "action", DOCUMENTS: "documents" },
      Action: { PICKED: "picked" },
      Document: {
        ID: "id",
        NAME: "name",
        MIME_TYPE: "mimeType",
        URL: "url",
        ICON_URL: "iconUrl",
      },
    },
  };

  return { build, setOAuthToken };
}

const toastMock = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  mockUseToast.mockReturnValue({ toast: toastMock });
  mockGetQueryOptions.mockReturnValue({
    queryKey: ["cred"],
    queryFn: () => Promise.resolve(undefined),
  });
});

afterEach(() => {
  const mockedWindow = window as unknown as {
    gapi?: unknown;
    google?: unknown;
  };
  delete mockedWindow.gapi;
  delete mockedWindow.google;
});

function renderPicker(
  overrides: Partial<Parameters<typeof GoogleDrivePicker>[0]> = {},
) {
  const onPicked = vi.fn();
  const onCanceled = vi.fn();
  const onError = vi.fn();
  const utils = render(
    <GoogleDrivePicker
      scopes={["drive.file", "drive.metadata"]}
      developerKey="dev-key"
      clientId="client-id"
      appId="app-id"
      onPicked={onPicked}
      onCanceled={onCanceled}
      onError={onError}
      {...overrides}
    />,
  );
  return { ...utils, onPicked, onCanceled, onError };
}

describe("GoogleDrivePicker – click-to-open flow", () => {
  it("shows insufficient scopes toast and calls onError when credential lacks required scopes", async () => {
    const { build } = setupGoogleGlobals();

    const savedCred = makeCred({ id: "narrow-cred", scopes: ["drive.file"] });

    mockUseCredentials.mockReturnValue({
      provider: "google",
      providerName: "Google",
      savedCredentials: [savedCred],
      upgradeableCredentials: [],
      supportsOAuth2: true,
      supportsApiKey: false,
      supportsUserPassword: false,
      supportsHostScoped: false,
      isLoading: false,
      isSystemProvider: false,
    });

    mockGetQueryOptions.mockReturnValue({
      queryKey: ["cred", "narrow-cred"],
      queryFn: () =>
        Promise.resolve({
          status: 200,
          data: { id: "narrow-cred", type: "oauth2", scopes: ["drive.file"] },
        }),
    });

    const { onError } = renderPicker();

    const button = await screen.findByRole("button", {
      name: /choose file\(s\) from google drive/i,
    });
    fireEvent.click(button);

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Insufficient Permissions",
          variant: "destructive",
        }),
      );
    });

    expect(onError).toHaveBeenCalledWith(expect.any(Error));
    expect(build).not.toHaveBeenCalled();
  });

  it("fetches picker token and builds picker when scopes are sufficient", async () => {
    const { build, setOAuthToken } = setupGoogleGlobals();

    const savedCred = makeCred({
      id: "full-cred",
      scopes: ["drive.file", "drive.metadata"],
    });

    mockUseCredentials.mockReturnValue({
      provider: "google",
      providerName: "Google",
      savedCredentials: [savedCred],
      upgradeableCredentials: [],
      supportsOAuth2: true,
      supportsApiKey: false,
      supportsUserPassword: false,
      supportsHostScoped: false,
      isLoading: false,
      isSystemProvider: false,
    });

    mockGetQueryOptions.mockReturnValue({
      queryKey: ["cred", "full-cred"],
      queryFn: () =>
        Promise.resolve({
          status: 200,
          data: {
            id: "full-cred",
            type: "oauth2",
            scopes: ["drive.file", "drive.metadata"],
          },
        }),
    });

    mockPostPickerToken.mockResolvedValue({
      status: 200,
      data: { access_token: "ya29.picker-token" },
    });

    const { onError } = renderPicker();

    const button = await screen.findByRole("button", {
      name: /choose file\(s\) from google drive/i,
    });
    fireEvent.click(button);

    await waitFor(() => {
      expect(mockPostPickerToken).toHaveBeenCalledWith("google", "full-cred");
    });
    expect(build).toHaveBeenCalled();
    expect(setOAuthToken).toHaveBeenCalledWith("ya29.picker-token");
    expect(onError).not.toHaveBeenCalled();
  });

  it("calls onError when stored credential is not oauth2", async () => {
    setupGoogleGlobals();

    const savedCred = makeCred({
      id: "api-key-cred",
      type: "api_key",
      scopes: [],
    });

    mockUseCredentials.mockReturnValue({
      provider: "google",
      providerName: "Google",
      savedCredentials: [savedCred],
      upgradeableCredentials: [],
      supportsOAuth2: true,
      supportsApiKey: true,
      supportsUserPassword: false,
      supportsHostScoped: false,
      isLoading: false,
      isSystemProvider: false,
    });

    mockGetQueryOptions.mockReturnValue({
      queryKey: ["cred", "api-key-cred"],
      queryFn: () =>
        Promise.resolve({
          status: 200,
          data: { id: "api-key-cred", type: "api_key" },
        }),
    });

    const { onError } = renderPicker({ scopes: ["drive.file"] });

    const button = await screen.findByRole("button", {
      name: /choose file\(s\) from google drive/i,
    });
    fireEvent.click(button);

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining("Failed to retrieve"),
        }),
      );
    });
  });
});
