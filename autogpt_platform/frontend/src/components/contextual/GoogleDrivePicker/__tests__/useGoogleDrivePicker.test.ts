import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
  okData: vi.fn((resp: any) => (resp?.status === 200 ? resp.data : undefined)),
}));

import useCredentials from "@/hooks/useCredentials";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  getGetV1GetSpecificCredentialByIdQueryOptions,
  postV1GetPickerToken,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useGoogleDrivePicker } from "../useGoogleDrivePicker";
import type { CredentialsMetaResponse } from "@/lib/autogpt-server-api";

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

// Minimal react-query QueryClient mock
const mockFetchQuery = vi.fn();
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ fetchQuery: mockFetchQuery }),
}));

// Mock Google globals for ensureLoaded + buildAndShowPicker
function setupGoogleGlobals() {
  const setOAuthToken = vi.fn();
  const setDeveloperKey = vi.fn();
  const setAppId = vi.fn();
  const setCallback = vi.fn();
  const enableFeature = vi.fn();
  const addView = vi.fn();
  const build = vi.fn().mockReturnValue({ setVisible: vi.fn() });

  // Use a class so `new PickerBuilder()` works properly in vitest 4.x
  class MockPickerBuilder {
    setOAuthToken = (...args: any[]) => {
      setOAuthToken(...args);
      return this;
    };
    setDeveloperKey = (...args: any[]) => {
      setDeveloperKey(...args);
      return this;
    };
    setAppId = (...args: any[]) => {
      setAppId(...args);
      return this;
    };
    setCallback = (...args: any[]) => {
      setCallback(...args);
      return this;
    };
    enableFeature = (...args: any[]) => {
      enableFeature(...args);
      return this;
    };
    addView = (...args: any[]) => {
      addView(...args);
      return this;
    };
    build = (...args: any[]) => build(...args);
  }

  class MockDocsView {
    setMode = vi.fn();
    constructor(_viewId?: any) {}
  }

  (window as any).gapi = {
    load: (_name: string, opts: { callback: () => void }) => opts.callback(),
  };
  (window as any).google = {
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
  mockGetQueryOptions.mockReturnValue({ queryKey: ["cred"], queryFn: vi.fn() });
});

afterEach(() => {
  delete (window as any).gapi;
  delete (window as any).google;
});

describe("useGoogleDrivePicker – openPicker saved-credential flow", () => {
  it("shows insufficient scopes toast when credential lacks required scopes", async () => {
    const { build } = setupGoogleGlobals();

    const savedCred = makeCred({
      id: "narrow-cred",
      scopes: ["drive.file"],
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

    // The credential returned by fetchQuery lacks the required scopes
    mockFetchQuery.mockResolvedValue({
      status: 200,
      data: {
        id: "narrow-cred",
        type: "oauth2",
        scopes: ["drive.file"],
      },
    });

    const onError = vi.fn();
    const onPicked = vi.fn();
    const onCanceled = vi.fn();

    const { result } = renderHook(() =>
      useGoogleDrivePicker({
        scopes: ["drive.file", "drive.metadata"],
        developerKey: "dev-key",
        clientId: "client-id",
        appId: "app-id",
        onPicked,
        onCanceled,
        onError,
      }),
    );

    await act(async () => {
      await result.current.handleOpenPicker();
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Insufficient Permissions",
        variant: "destructive",
      }),
    );
    expect(onError).toHaveBeenCalledWith(expect.any(Error));
    // Picker should NOT have been built
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

    // fetchQuery returns oauth2 credential with all scopes
    mockFetchQuery.mockResolvedValue({
      status: 200,
      data: {
        id: "full-cred",
        type: "oauth2",
        scopes: ["drive.file", "drive.metadata"],
      },
    });

    // Mock picker token endpoint
    mockPostPickerToken.mockResolvedValue({
      status: 200,
      data: { access_token: "ya29.picker-token" },
    });

    const onError = vi.fn();
    const onPicked = vi.fn();
    const onCanceled = vi.fn();

    const { result } = renderHook(() =>
      useGoogleDrivePicker({
        scopes: ["drive.file", "drive.metadata"],
        developerKey: "dev-key",
        clientId: "client-id",
        appId: "app-id",
        onPicked,
        onCanceled,
        onError,
      }),
    );

    await act(async () => {
      await result.current.handleOpenPicker();
    });

    // Should have fetched a picker token
    expect(mockPostPickerToken).toHaveBeenCalledWith("google", "full-cred");
    // Picker should have been built with the token
    expect(build).toHaveBeenCalled();
    expect(setOAuthToken).toHaveBeenCalledWith("ya29.picker-token");
    expect(onError).not.toHaveBeenCalled();
  });

  it("calls onError when credential is not oauth2", async () => {
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

    mockFetchQuery.mockResolvedValue({
      status: 200,
      data: { id: "api-key-cred", type: "api_key" },
    });

    const onError = vi.fn();
    const onPicked = vi.fn();
    const onCanceled = vi.fn();

    const { result } = renderHook(() =>
      useGoogleDrivePicker({
        scopes: ["drive.file"],
        developerKey: "dev-key",
        clientId: "client-id",
        appId: "app-id",
        onPicked,
        onCanceled,
        onError,
      }),
    );

    await act(async () => {
      await result.current.handleOpenPicker();
    });

    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Failed to retrieve"),
      }),
    );
  });
});
