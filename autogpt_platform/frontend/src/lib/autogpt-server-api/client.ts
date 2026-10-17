import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import { getWebSocketToken } from "@/lib/auth/actions";
import { ImpersonationState } from "@/lib/impersonation";
import { getDatafastAttribution } from "@/services/analytics/datafast-attribution";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";
import type {
  AddUserCreditsResponse,
  APIKey,
  APIKeyCredentials,
  APIKeyPermission,
  Block,
  CreateAPIKeyResponse,
  Credentials,
  CredentialsDeleteNeedConfirmationResponse,
  CredentialsDeleteResponse,
  CredentialsMetaInput,
  CredentialsMetaResponse,
  Graph,
  GraphCreatable,
  GraphExecution,
  GraphExecutionID,
  GraphExecutionMeta,
  GraphExecutionsResponse,
  GraphID,
  GraphMeta,
  GraphUpdateable,
  HostScopedCredentials,
  LibraryAgent,
  LibraryAgentID,
  LibraryAgentPreset,
  LibraryAgentPresetCreatable,
  LibraryAgentPresetCreatableFromGraphExecution,
  LibraryAgentPresetID,
  LibraryAgentPresetResponse,
  LibraryAgentPresetUpdatable,
  LibraryAgentResponse,
  LibraryAgentSortEnum,
  NodeExecutionResult,
  NotificationPreference,
  NotificationPreferenceDTO,
  OttoQuery,
  OttoResponse,
  RefundRequest,
  Schedule,
  ScheduleCreatable,
  ScheduleID,
  SkippedWebhookPreset,
  TransactionHistory,
  User,
  UserPasswordCredentials,
  UsersBalanceHistoryResponse,
  WebSocketNotification,
} from "./types";
import { BackendWebSocket } from "./websocket-connection";

const isClient = environment.isClientSide();

/**
 * Thrown when a request fails because the user is logging out.
 * Callers can catch this specifically to silently ignore logout-related failures,
 * rather than receiving null and crashing on property access.
 */
export class LogoutInterruptError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LogoutInterruptError";
  }
}

/**
 * Build the query object for `oAuthLogin`.  Kept as a named helper so the
 * shape — scopes-only vs credential_id-only vs both vs neither — can be
 * pinned in tests without mocking the whole BackendAPI request layer.
 *
 * Returns `undefined` when neither argument is provided so callers can
 * omit the query string entirely.
 */
export function buildOAuthLoginQuery(
  scopes?: string[],
  credentialID?: string,
): Record<string, string> | undefined {
  const query: Record<string, string> = {};
  if (scopes && scopes.length > 0) query.scopes = scopes.join(",");
  if (credentialID) query.credential_id = credentialID;
  return Object.keys(query).length > 0 ? query : undefined;
}

export default class BackendAPI {
  private baseUrl: string;
  private webSocket: BackendWebSocket;
  private wsOnConnectHandlers: Set<() => void> = new Set();
  private wsOnDisconnectHandlers: Set<() => void> = new Set();
  private wsMessageHandlers: Record<string, Set<(data: any) => void>> = {};
  constructor(
    baseUrl: string = environment.getAGPTServerApiUrl(),
    wsUrl: string = environment.getAGPTWsServerUrl(),
  ) {
    this.baseUrl = baseUrl;
    this.webSocket = new BackendWebSocket(wsUrl, {
      onConnect: () => this.wsOnConnectHandlers.forEach((handler) => handler()),
      onDisconnect: () =>
        this.wsOnDisconnectHandlers.forEach((handler) => handler()),
      onMessage: (event) => this._handleWSMessage(event),
    });
  }

  async isAuthenticated(): Promise<boolean> {
    // getWebSocketToken is a server action, so this works from both the
    // client (RPC) and the server (direct call); a token is only minted
    // when a valid session exists.
    const { token, error } = await getWebSocketToken();
    return token != null && !error;
  }

  createUser(): Promise<User> {
    return this._request("POST", "/auth/user", {});
  }

  updateUserEmail(email: string): Promise<{ email: string }> {
    return this._request("POST", "/auth/user/email", { email });
  }

  ////////////////////////////////////////
  /////////////// CREDITS ////////////////
  ////////////////////////////////////////

  async getUserCredit(): Promise<{ credits: number | null }> {
    try {
      const response = await this._get("/credits");
      return response ?? { credits: 0 };
    } catch (error) {
      if (!(error instanceof LogoutInterruptError)) {
        Sentry.captureException(error);
      }
      // Return null (rather than 0) so callers can distinguish a real $0
      // balance from a failed fetch — used by TopUpPromptProvider to avoid
      // nudging users to top up on transient API errors.
      return { credits: null };
    }
  }

  getUserPreferences(): Promise<NotificationPreferenceDTO> {
    return this._get("/auth/user/preferences");
  }

  updateUserPreferences(
    preferences: NotificationPreferenceDTO,
  ): Promise<NotificationPreference> {
    return this._request("POST", "/auth/user/preferences", preferences);
  }

  getAutoTopUpConfig(): Promise<{ amount: number; threshold: number }> {
    return this._get("/credits/auto-top-up");
  }

  setAutoTopUpConfig(config: {
    amount: number;
    threshold: number;
  }): Promise<{ amount: number; threshold: number }> {
    return this._request("POST", "/credits/auto-top-up", config);
  }

  getTransactionHistory(
    lastTransction: Date | null = null,
    countLimit: number | null = null,
    transactionType: string | null = null,
  ): Promise<TransactionHistory> {
    const filters: Record<string, any> = {};
    if (lastTransction) filters.transaction_time = lastTransction;
    if (countLimit) filters.transaction_count_limit = countLimit;
    if (transactionType) filters.transaction_type = transactionType;
    return this._get(`/credits/transactions`, filters);
  }

  getRefundRequests(): Promise<RefundRequest[]> {
    return this._get(`/credits/refunds`);
  }

  requestTopUp(credit_amount: number): Promise<{ checkout_url: string }> {
    return this._request("POST", "/credits", { credit_amount });
  }

  refundTopUp(transaction_key: string, reason: string): Promise<number> {
    return this._request("POST", `/credits/${transaction_key}/refund`, {
      reason,
    });
  }

  getUserPaymentPortalLink(): Promise<{ url: string }> {
    return this._get("/credits/manage");
  }

  fulfillCheckout(): Promise<void> {
    return this._request("PATCH", "/credits");
  }

  ////////////////////////////////////////
  //////////////// GRAPHS ////////////////
  ////////////////////////////////////////

  getBlocks(): Promise<Block[]> {
    return this._get("/blocks");
  }

  listGraphs(): Promise<GraphMeta[]> {
    return this._get(`/graphs`);
  }

  async getGraph(
    id: GraphID,
    version?: number,
    for_export?: boolean,
  ): Promise<Graph> {
    const query: Record<string, any> = {};
    if (version !== undefined) {
      query["version"] = version;
    }
    if (for_export !== undefined) {
      query["for_export"] = for_export;
    }
    const graph = await this._get(`/graphs/${id}`, query);
    if (for_export) delete graph.user_id;
    return graph;
  }

  getGraphAllVersions(id: GraphID): Promise<Graph[]> {
    return this._get(`/graphs/${id}/versions`);
  }

  createGraph(
    graph: GraphCreatable,
    source?: GraphCreationSource,
  ): Promise<Graph> {
    const requestBody: GraphCreateRequestBody = { graph };
    if (source) {
      requestBody.source = source;
    }

    return this._request("POST", "/graphs", requestBody);
  }

  updateGraph(
    id: GraphID,
    graph: GraphUpdateable,
  ): Promise<{
    graph: Graph;
    skipped_webhook_presets?: SkippedWebhookPreset[];
  }> {
    return this._request("PUT", `/graphs/${id}`, graph);
  }

  deleteGraph(id: GraphID): Promise<void> {
    return this._request("DELETE", `/graphs/${id}`);
  }

  setGraphActiveVersion(
    id: GraphID,
    version: number,
  ): Promise<{ skipped_webhook_presets?: SkippedWebhookPreset[] }> {
    return this._request("PUT", `/graphs/${id}/versions/active`, {
      active_graph_version: version,
    });
  }

  executeGraph(
    id: GraphID,
    version: number,
    inputs: { [key: string]: any } = {},
    credentials_inputs: { [key: string]: CredentialsMetaInput } = {},
    source?: GraphExecutionSource,
  ): Promise<GraphExecutionMeta> {
    const body: GraphExecuteRequestBody = { inputs, credentials_inputs };
    if (source) {
      body.source = source;
    }
    return this._request("POST", `/graphs/${id}/execute/${version}`, body);
  }

  getExecutions(): Promise<GraphExecutionMeta[]> {
    return this._get(`/executions`).then((results) =>
      results.map(parseGraphExecutionTimestamps),
    );
  }

  getGraphExecutions(graphID: GraphID): Promise<GraphExecutionsResponse> {
    return this._get(`/graphs/${graphID}/executions`).then((results) =>
      results.map(parseGraphExecutionTimestamps),
    );
  }

  async getGraphExecutionInfo(
    graphID: GraphID,
    runID: GraphExecutionID,
  ): Promise<GraphExecution> {
    const result = await this._get(`/graphs/${graphID}/executions/${runID}`);
    return parseGraphExecutionTimestamps<GraphExecution>(result);
  }

  async stopGraphExecution(
    graphID: GraphID,
    runID: GraphExecutionID,
  ): Promise<GraphExecution> {
    const result = await this._request(
      "POST",
      `/graphs/${graphID}/executions/${runID}/stop`,
    );
    return parseGraphExecutionTimestamps<GraphExecution>(result);
  }

  async deleteGraphExecution(runID: GraphExecutionID): Promise<void> {
    await this._request("DELETE", `/executions/${runID}`);
  }

  oAuthLogin(
    provider: string,
    scopes?: string[],
    credentialID?: string,
  ): Promise<{
    login_url: string;
    state_token: string;
    cancel_url?: string | null;
  }> {
    return this._get(
      `/integrations/${provider}/login`,
      buildOAuthLoginQuery(scopes, credentialID),
    );
  }

  oAuthCallback(
    provider: string,
    code: string,
    state_token: string,
  ): Promise<CredentialsMetaResponse> {
    return this._request("POST", `/integrations/${provider}/callback`, {
      code,
      state_token,
    });
  }

  createAPIKeyCredentials(
    credentials: Omit<APIKeyCredentials, "id" | "type">,
  ): Promise<APIKeyCredentials> {
    return this._request(
      "POST",
      `/integrations/${credentials.provider}/credentials`,
      { ...credentials, type: "api_key" },
    );
  }

  createUserPasswordCredentials(
    credentials: Omit<UserPasswordCredentials, "id" | "type">,
  ): Promise<UserPasswordCredentials> {
    return this._request(
      "POST",
      `/integrations/${credentials.provider}/credentials`,
      { ...credentials, type: "user_password" },
    );
  }

  createHostScopedCredentials(
    credentials: Omit<HostScopedCredentials, "id" | "type">,
  ): Promise<HostScopedCredentials> {
    return this._request(
      "POST",
      `/integrations/${credentials.provider}/credentials`,
      { ...credentials, type: "host_scoped" },
    );
  }

  async listProviders(): Promise<string[]> {
    // The endpoint returns `ProviderMetadata[]` (`{ name, description }`) but
    // legacy consumers (e.g. CredentialsProvider) still expect a flat string[]
    // of provider names. Map down so we keep that contract.
    const response: Array<{ name: string }> = await this._get(
      "/integrations/providers",
    );
    return response.map((p) => p.name);
  }

  listSystemProviders(): Promise<string[]> {
    return this._get("/integrations/providers/system");
  }

  listCredentials(provider?: string): Promise<CredentialsMetaResponse[]> {
    return this._get(
      provider
        ? `/integrations/${provider}/credentials`
        : "/integrations/credentials",
    );
  }

  getCredentials(provider: string, id: string): Promise<Credentials> {
    return this._get(`/integrations/${provider}/credentials/${id}`);
  }

  deleteCredentials(
    provider: string,
    id: string,
    force: boolean = true,
  ): Promise<
    CredentialsDeleteResponse | CredentialsDeleteNeedConfirmationResponse
  > {
    return this._request(
      "DELETE",
      `/integrations/${provider}/credentials/${id}`,
      force ? { force: true } : undefined,
    );
  }

  // API Key related requests
  async createAPIKey(
    name: string,
    permissions: APIKeyPermission[],
    description?: string,
  ): Promise<CreateAPIKeyResponse> {
    return this._request("POST", "/api-keys", {
      name,
      permissions,
      description,
    });
  }

  async listAPIKeys(): Promise<APIKey[]> {
    return this._get("/api-keys");
  }

  async revokeAPIKey(keyId: string): Promise<APIKey> {
    return this._request("DELETE", `/api-keys/${keyId}`);
  }

  async updateAPIKeyPermissions(
    keyId: string,
    permissions: APIKeyPermission[],
  ): Promise<APIKey> {
    return this._request("PUT", `/api-keys/${keyId}/permissions`, {
      permissions,
    });
  }

  /**
   * @returns `true` if a ping event was received, `false` if provider doesn't support pinging but the webhook exists.
   * @throws  `Error` if the webhook does not exist.
   * @throws  `Error` if the attempt to ping timed out.
   */
  async pingWebhook(webhook_id: string): Promise<boolean> {
    return this._request("POST", `/integrations/webhooks/${webhook_id}/ping`);
  }

  async uploadFile(
    file: File,
    expiration_hours: number = 24,
    onProgress?: (progress: number) => void,
  ): Promise<{
    file_uri: string;
    file_name: string;
    size: number;
    content_type: string;
    expires_in_hours: number;
  }> {
    const response = await this._uploadFileWithProgress(
      "/files/upload",
      file,
      { expiration_hours },
      onProgress,
    );
    if (typeof response === "string") {
      return JSON.parse(response);
    }
    return response;
  }

  /////////////////////////////////////////
  /////////// Admin API ///////////////////
  /////////////////////////////////////////

  addUserCredits(
    user_id: string,
    amount: number,
    comments: string,
  ): Promise<AddUserCreditsResponse> {
    return this._request("POST", "/credits/admin/add_credits", {
      user_id,
      amount,
      comments,
    });
  }

  getUsersHistory(params?: {
    search?: string;
    page?: number;
    page_size?: number;
    transaction_filter?: string;
  }): Promise<UsersBalanceHistoryResponse> {
    return this._get("/credits/admin/users_history", params);
  }

  ////////////////////////////////////////
  //////////// V2 LIBRARY API ////////////
  ////////////////////////////////////////

  listLibraryAgents(params?: {
    search_term?: string;
    sort_by?: LibraryAgentSortEnum;
    page?: number;
    page_size?: number;
  }): Promise<LibraryAgentResponse> {
    return this._get("/library/agents", params);
  }

  listFavoriteLibraryAgents(params?: {
    page?: number;
    page_size?: number;
  }): Promise<LibraryAgentResponse> {
    return this._get("/library/agents/favorites", params);
  }

  getLibraryAgent(id: LibraryAgentID): Promise<LibraryAgent> {
    return this._get(`/library/agents/${id}`);
  }

  getLibraryAgentByStoreListingVersionID(
    storeListingVersionId: string,
  ): Promise<LibraryAgent | null> {
    return this._get(`/library/agents/marketplace/${storeListingVersionId}`);
  }

  getLibraryAgentByGraphID(
    graphID: GraphID,
    graphVersion?: number,
  ): Promise<LibraryAgent> {
    return this._get(`/library/agents/by-graph/${graphID}`, {
      version: graphVersion,
    });
  }

  updateLibraryAgent(
    libraryAgentId: LibraryAgentID,
    params: {
      auto_update_version?: boolean;
      is_favorite?: boolean;
      is_archived?: boolean;
    },
  ): Promise<LibraryAgent> {
    return this._request("PATCH", `/library/agents/${libraryAgentId}`, params);
  }

  async deleteLibraryAgent(libraryAgentId: LibraryAgentID): Promise<void> {
    await this._request("DELETE", `/library/agents/${libraryAgentId}`);
  }

  forkLibraryAgent(libraryAgentId: LibraryAgentID): Promise<LibraryAgent> {
    return this._request("POST", `/library/agents/${libraryAgentId}/fork`);
  }

  async setupAgentTrigger(params: {
    name: string;
    description?: string;
    graph_id: GraphID;
    graph_version: number;
    trigger_config: Record<string, any>;
    agent_credentials: Record<string, CredentialsMetaInput>;
  }): Promise<LibraryAgentPreset> {
    return parseLibraryAgentPresetTimestamp(
      await this._request("POST", `/library/presets/setup-trigger`, params),
    );
  }

  async listLibraryAgentPresets(params?: {
    graph_id?: GraphID;
    page?: number;
    page_size?: number;
  }): Promise<LibraryAgentPresetResponse> {
    const response: LibraryAgentPresetResponse = await this._get(
      "/library/presets",
      params,
    );
    return {
      ...response,
      presets: response.presets.map(parseLibraryAgentPresetTimestamp),
    };
  }

  async getLibraryAgentPreset(
    presetID: LibraryAgentPresetID,
  ): Promise<LibraryAgentPreset> {
    const preset = await this._get(`/library/presets/${presetID}`);
    return parseLibraryAgentPresetTimestamp(preset);
  }

  async createLibraryAgentPreset(
    params:
      | LibraryAgentPresetCreatable
      | LibraryAgentPresetCreatableFromGraphExecution,
  ): Promise<LibraryAgentPreset> {
    const new_preset = await this._request("POST", "/library/presets", params);
    return parseLibraryAgentPresetTimestamp(new_preset);
  }

  async updateLibraryAgentPreset(
    presetID: LibraryAgentPresetID,
    partial_preset: LibraryAgentPresetUpdatable,
  ): Promise<LibraryAgentPreset> {
    const updated_preset = await this._request(
      "PATCH",
      `/library/presets/${presetID}`,
      partial_preset,
    );
    return parseLibraryAgentPresetTimestamp(updated_preset);
  }

  async deleteLibraryAgentPreset(
    presetID: LibraryAgentPresetID,
  ): Promise<void> {
    await this._request("DELETE", `/library/presets/${presetID}`);
  }

  executeLibraryAgentPreset(
    presetID: LibraryAgentPresetID,
    inputs?: Record<string, any>,
    credential_inputs?: Record<string, CredentialsMetaInput>,
  ): Promise<GraphExecutionMeta> {
    return this._request("POST", `/library/presets/${presetID}/execute`, {
      inputs,
      credential_inputs,
    });
  }

  //////////////////////////////////
  /////////// SCHEDULES ////////////
  //////////////////////////////////

  async createGraphExecutionSchedule(
    params: ScheduleCreatable,
  ): Promise<Schedule> {
    return this._request(
      "POST",
      `/graphs/${params.graph_id}/schedules`,
      params,
    ).then(parseScheduleTimestamp);
  }

  async listGraphExecutionSchedules(graphID: GraphID): Promise<Schedule[]> {
    return this._get(`/graphs/${graphID}/schedules`).then((schedules) =>
      schedules.map(parseScheduleTimestamp),
    );
  }

  /** @deprecated only used in legacy `Monitor` */
  async listAllGraphsExecutionSchedules(): Promise<Schedule[]> {
    return this._get(`/schedules`).then((schedules) =>
      schedules.map(parseScheduleTimestamp),
    );
  }

  async deleteGraphExecutionSchedule(
    scheduleID: ScheduleID,
  ): Promise<{ id: ScheduleID }> {
    return this._request("DELETE", `/schedules/${scheduleID}`);
  }

  //////////////////////////////////
  ////////////// OTTO //////////////
  //////////////////////////////////

  async askOtto(query: OttoQuery): Promise<OttoResponse> {
    return this._request("POST", "/otto/ask", query);
  }

  ////////////////////////////////////////
  ////////// INTERNAL FUNCTIONS //////////
  ////////////////////////////////////////

  private _get(path: string, query?: Record<string, any>) {
    return this._request("GET", path, query);
  }

  private async _uploadFile(path: string, file: File): Promise<string> {
    const formData = new FormData();
    formData.append("file", file);

    if (isClient) {
      return this._makeClientFileUpload(path, formData);
    } else {
      return this._makeServerFileUpload(path, formData);
    }
  }

  private async _uploadFileWithProgress(
    path: string,
    file: File,
    params?: Record<string, any>,
    onProgress?: (progress: number) => void,
  ): Promise<string> {
    const formData = new FormData();
    formData.append("file", file);

    if (isClient) {
      return this._makeClientFileUploadWithProgress(
        path,
        formData,
        params,
        onProgress,
      );
    } else {
      return this._makeServerFileUploadWithProgress(path, formData, params);
    }
  }

  private async _makeClientFileUpload(
    path: string,
    formData: FormData,
  ): Promise<string> {
    // Dynamic import is required even for client-only functions because helpers.ts
    // has server-only imports (like getBackendAuthToken) at the top level. Static imports
    // would bundle server-only code into the client bundle, causing runtime errors.
    const { buildClientUrl, handleFetchError } = await import("./helpers");

    const uploadUrl = buildClientUrl(path);

    const response = await fetch(uploadUrl, {
      method: "POST",
      body: formData,
      credentials: "include",
    });

    if (!response.ok) {
      throw await handleFetchError(response);
    }

    return await response.json();
  }

  private async _makeServerFileUpload(
    path: string,
    formData: FormData,
  ): Promise<string> {
    const { makeAuthenticatedFileUpload, buildServerUrl } = await import(
      "./helpers"
    );
    const url = buildServerUrl(path);
    return await makeAuthenticatedFileUpload(url, formData);
  }

  private async _makeClientFileUploadWithProgress(
    path: string,
    formData: FormData,
    params?: Record<string, any>,
    onProgress?: (progress: number) => void,
  ): Promise<any> {
    const { buildClientUrl, buildUrlWithQuery } = await import("./helpers");

    let url = buildClientUrl(path);
    if (params) {
      url = buildUrlWithQuery(url, params);
    }

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      if (onProgress) {
        xhr.upload.addEventListener("progress", (e) => {
          if (e.lengthComputable) {
            const progress = (e.loaded / e.total) * 100;
            onProgress(progress);
          }
        });
      }

      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve(response);
          } catch (_error) {
            reject(new Error("Invalid JSON response"));
          }
        } else {
          // Handle file size errors with user-friendly message
          if (xhr.status === 413) {
            reject(new Error("File is too large — max size is 256MB"));
            return;
          }

          // Try to parse error response for better messages
          let errorMessage = `Upload failed (${xhr.status})`;
          try {
            const errorData = JSON.parse(xhr.responseText);
            if (errorData.detail) {
              if (
                typeof errorData.detail === "string" &&
                errorData.detail.includes("exceeds the maximum")
              ) {
                const match = errorData.detail.match(
                  /maximum allowed size of (\d+)MB/,
                );
                const maxSize = match ? match[1] : "256";
                errorMessage = `File is too large — max size is ${maxSize}MB`;
              } else if (typeof errorData.detail === "string") {
                errorMessage = errorData.detail;
              }
            } else if (errorData.error) {
              errorMessage = errorData.error;
            }
          } catch {
            // Keep default message if parsing fails
          }

          reject(new Error(errorMessage));
        }
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Network error"));
      });

      xhr.open("POST", url);
      xhr.withCredentials = true;
      xhr.send(formData);
    });
  }

  private async _makeServerFileUploadWithProgress(
    path: string,
    formData: FormData,
    params?: Record<string, any>,
  ): Promise<string> {
    const { makeAuthenticatedFileUpload, buildServerUrl, buildUrlWithQuery } =
      await import("./helpers");

    let url = buildServerUrl(path);
    if (params) {
      url = buildUrlWithQuery(url, params);
    }

    return await makeAuthenticatedFileUpload(url, formData);
  }

  private async _request(
    method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
    path: string,
    payload?: Record<string, any>,
  ) {
    if (method !== "GET") {
      console.debug(`${method} ${path} payload:`, payload);
    }

    if (isClient) {
      return this._makeClientRequest(method, path, payload);
    } else {
      return this._makeServerRequest(method, path, payload);
    }
  }

  private async _makeClientRequest(
    method: string,
    path: string,
    payload?: Record<string, any>,
  ) {
    // Dynamic import is required even for client-only functions because helpers.ts
    // has server-only imports (like getBackendAuthToken) at the top level. Static imports
    // would bundle server-only code into the client bundle, causing runtime errors.
    const {
      buildClientUrl,
      buildUrlWithQuery,
      handleFetchError,
      isLogoutInProgress,
      isAuthenticationError,
    } = await import("./helpers");

    const payloadAsQuery = ["GET", "DELETE"].includes(method);
    let url = buildClientUrl(path);

    if (payloadAsQuery && payload) {
      url = buildUrlWithQuery(url, payload);
    }

    // Prepare headers with admin impersonation support
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    const impersonatedUserId = ImpersonationState.get();
    if (impersonatedUserId) {
      headers[IMPERSONATION_HEADER_NAME] = impersonatedUserId;
    }

    Object.assign(headers, getDatafastAttribution());

    const response = await fetch(url, {
      method,
      headers,
      body: !payloadAsQuery && payload ? JSON.stringify(payload) : undefined,
      credentials: "include",
    });

    if (!response.ok) {
      const error = await handleFetchError(response);
      if (
        isAuthenticationError(response, error.message) &&
        isLogoutInProgress()
      ) {
        console.debug(
          "Authentication request failed during logout, ignoring:",
          error.message,
        );
        throw new LogoutInterruptError("Request cancelled: logout in progress");
      }
      throw error;
    }

    if (
      response.status === 204 ||
      response.headers.get("Content-Length") === "0"
    ) {
      return null;
    }
    return await response.json();
  }

  private async _makeServerRequest(
    method: string,
    path: string,
    payload?: Record<string, any>,
  ) {
    const { makeAuthenticatedRequest, buildServerUrl } = await import(
      "./helpers"
    );
    const url = buildServerUrl(path);

    // For server-side requests, try to read impersonation from cookies
    const impersonationUserId = await ImpersonationState.getServerSide();
    const fakeRequest = impersonationUserId
      ? new Request(url, {
          headers: { "X-Act-As-User-Id": impersonationUserId },
        })
      : undefined;

    return await makeAuthenticatedRequest(
      method,
      url,
      payload,
      "application/json",
      fakeRequest,
    );
  }

  ////////////////////////////////////////
  ////////////// WEBSOCKETS //////////////
  ////////////////////////////////////////

  subscribeToGraphExecution(graphExecID: GraphExecutionID): Promise<void> {
    return this.sendWebSocketMessage("subscribe_graph_execution", {
      graph_exec_id: graphExecID,
    });
  }

  subscribeToGraphExecutions(graphID: GraphID): Promise<void> {
    return this.sendWebSocketMessage("subscribe_graph_executions", {
      graph_id: graphID,
    });
  }

  async sendWebSocketMessage<M extends keyof WebsocketMessageTypeMap>(
    method: M,
    data: WebsocketMessageTypeMap[M],
  ): Promise<void> {
    const message = JSON.stringify({ method, data });
    if (this.webSocket.send(message)) return;
    await this.connectWebSocket();
    if (!this.webSocket.send(message)) {
      throw new Error("WebSocket connection is not authenticated");
    }
  }

  onWebSocketMessage<M extends keyof WebsocketMessageTypeMap>(
    method: M,
    handler: (data: WebsocketMessageTypeMap[M]) => void,
  ): () => void {
    this.wsMessageHandlers[method] ??= new Set();
    this.wsMessageHandlers[method].add(handler);

    // Return detacher
    return () => this.wsMessageHandlers[method].delete(handler);
  }

  /**
   * All handlers are invoked when the WebSocket (re)connects. If it's already connected
   * when this function is called, the passed handler is invoked immediately.
   *
   * Use this hook to subscribe to topics and refresh state,
   * to ensure re-subscription and re-sync on re-connect.
   *
   * @returns a detacher for the passed handler.
   */
  onWebSocketConnect(handler: () => void): () => void {
    this.wsOnConnectHandlers.add(handler);

    if (this.webSocket.connected) handler();
    else void this.connectWebSocket();

    // Return detacher
    return () => this.wsOnConnectHandlers.delete(handler);
  }

  /**
   * All handlers are invoked when the WebSocket disconnects.
   *
   * @returns a detacher for the passed handler.
   */
  onWebSocketDisconnect(handler: () => void): () => void {
    this.wsOnDisconnectHandlers.add(handler);

    // Return detacher
    return () => this.wsOnDisconnectHandlers.delete(handler);
  }

  connectWebSocket(): Promise<void> {
    return this.webSocket.connect();
  }

  disconnectWebSocket() {
    this.webSocket.disconnect();
  }

  private _handleWSMessage(event: MessageEvent): void {
    const message: WebsocketMessage = JSON.parse(event.data);

    if (message.method === "node_execution_event") {
      message.data = parseNodeExecutionResultTimestamps(message.data);
    } else if (message.method == "graph_execution_event") {
      message.data = parseGraphExecutionTimestamps(message.data);
    }
    this.wsMessageHandlers[message.method]?.forEach((handler) =>
      handler(message.data),
    );
  }
}

/* *** UTILITY TYPES *** */

type GraphCreationSource = "builder" | "upload";
type GraphExecutionSource = "builder" | "library" | "onboarding";

type GraphCreateRequestBody = {
  graph: GraphCreatable;
  source?: GraphCreationSource;
};

type GraphExecuteRequestBody = {
  inputs: { [key: string]: any };
  credentials_inputs: { [key: string]: CredentialsMetaInput };
  source?: GraphExecutionSource;
};

type WebsocketMessageTypeMap = {
  subscribe_graph_execution: { graph_exec_id: GraphExecutionID };
  subscribe_graph_executions: { graph_id: GraphID };
  graph_execution_event: GraphExecution;
  node_execution_event: NodeExecutionResult;
  notification: WebSocketNotification;
  heartbeat: "ping" | "pong";
};

type WebsocketMessage = {
  [M in keyof WebsocketMessageTypeMap]: {
    method: M;
    data: WebsocketMessageTypeMap[M];
  };
}[keyof WebsocketMessageTypeMap];

type _PydanticValidationError = {
  type: string;
  loc: string[];
  msg: string;
  input: any;
};

/* *** HELPER FUNCTIONS *** */

function parseGraphExecutionTimestamps<
  T extends GraphExecutionMeta | GraphExecution,
>(result: any): T {
  const fixed = _parseObjectTimestamps<T>(result, ["started_at", "ended_at"]);
  if ("node_executions" in fixed && fixed.node_executions) {
    fixed.node_executions = fixed.node_executions.map(
      parseNodeExecutionResultTimestamps,
    );
  }
  return fixed;
}

function parseNodeExecutionResultTimestamps(result: any): NodeExecutionResult {
  return _parseObjectTimestamps<NodeExecutionResult>(result, [
    "add_time",
    "queue_time",
    "start_time",
    "end_time",
  ]);
}

function parseScheduleTimestamp(result: any): Schedule {
  return _parseObjectTimestamps<Schedule>(result, ["next_run_time"]);
}

function parseLibraryAgentPresetTimestamp(result: any): LibraryAgentPreset {
  return _parseObjectTimestamps<LibraryAgentPreset>(result, ["updated_at"]);
}

function _parseObjectTimestamps<T>(obj: any, keys: (keyof T)[]): T {
  const result = { ...obj };
  keys.forEach(
    (key) => (result[key] = result[key] ? new Date(result[key]) : undefined),
  );
  return result;
}
