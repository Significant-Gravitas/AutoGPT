import { getWebSocketToken } from "@/lib/supabase/actions";
import { getServerSupabase } from "@/lib/supabase/server/getServerSupabase";
import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";
import { Key, storage } from "@/services/storage/local-storage";
import {
  IMPERSONATION_HEADER_NAME,
  IMPERSONATION_STORAGE_KEY,
} from "@/lib/constants";
import * as Sentry from "@sentry/nextjs";
import type {
  AddUserCreditsResponse,
  AnalyticsDetails,
  AnalyticsMetrics,
  APIKey,
  APIKeyCredentials,
  APIKeyPermission,
  Block,
  CreateAPIKeyResponse,
  CreatorDetails,
  CreatorsResponse,
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
  MyAgentsResponse,
  NodeExecutionResult,
  NotificationPreference,
  NotificationPreferenceDTO,
  OttoQuery,
  OttoResponse,
  ProfileDetails,
  RefundRequest,
  ReviewSubmissionRequest,
  Schedule,
  ScheduleCreatable,
  ScheduleID,
  StoreAgentDetails,
  StoreAgentsResponse,
  StoreListingsWithVersionsResponse,
  StoreReview,
  StoreReviewCreate,
  StoreSubmission,
  StoreSubmissionRequest,
  StoreSubmissionsResponse,
  SubmissionStatus,
  TransactionHistory,
  User,
  UserOnboarding,
  UserPasswordCredentials,
  UsersBalanceHistoryResponse,
} from "./types";
import { environment } from "@/services/environment";

const isClient = environment.isClientSide();

export default class BackendAPI {
  private baseUrl: string;
  private wsUrl: string;
  private webSocket: WebSocket | null = null;
  private wsConnecting: Promise<void> | null = null;
  private wsOnConnectHandlers: Set<() => void> = new Set();
  private wsOnDisconnectHandlers: Set<() => void> = new Set();
  private wsMessageHandlers: Record<string, Set<(data: any) => void>> = {};
  private isIntentionallyDisconnected: boolean = false;

  readonly HEARTBEAT_INTERVAL = 100_000; // 100 seconds
  readonly HEARTBEAT_TIMEOUT = 10_000; // 10 seconds
  heartbeatIntervalID: number | null = null;
  heartbeatTimeoutID: number | null = null;

  constructor(
    baseUrl: string = environment.getAGPTServerApiUrl(),
    wsUrl: string = environment.getAGPTWsServerUrl(),
  ) {
    this.baseUrl = baseUrl;
    this.wsUrl = wsUrl;
  }

  private async getSupabaseClient(): Promise<SupabaseClient | null> {
    return isClient
      ? createBrowserClient(
          environment.getSupabaseUrl(),
          environment.getSupabaseAnonKey(),
          {
            isSingleton: true,
          },
        )
      : await getServerSupabase();
  }

  async isAuthenticated(): Promise<boolean> {
    const supabaseClient = await this.getSupabaseClient();
    if (!supabaseClient) return false;
    const {
      data: { session },
    } = await supabaseClient.auth.getSession();
    return session != null;
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

  getUserCredit(): Promise<{ credits: number }> {
    try {
      return this._get("/credits");
    } catch {
      return Promise.resolve({ credits: 0 });
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
  ////////////// ONBOARDING //////////////
  ////////////////////////////////////////

  getUserOnboarding(): Promise<UserOnboarding> {
    return this._get("/onboarding");
  }

  updateUserOnboarding(
    onboarding: Omit<Partial<UserOnboarding>, "rewardedFor">,
  ): Promise<void> {
    return this._request("PATCH", "/onboarding", onboarding);
  }

  getOnboardingAgents(): Promise<StoreAgentDetails[]> {
    return this._get("/onboarding/agents");
  }

  /** Check if onboarding is enabled not if user finished it or not. */
  isOnboardingEnabled(): Promise<boolean> {
    return this._get("/onboarding/enabled");
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

  createGraph(graph: GraphCreatable): Promise<Graph> {
    const requestBody = { graph } as GraphCreateRequestBody;

    return this._request("POST", "/graphs", requestBody);
  }

  updateGraph(id: GraphID, graph: GraphUpdateable): Promise<Graph> {
    return this._request("PUT", `/graphs/${id}`, graph);
  }

  deleteGraph(id: GraphID): Promise<void> {
    return this._request("DELETE", `/graphs/${id}`);
  }

  setGraphActiveVersion(id: GraphID, version: number): Promise<Graph> {
    return this._request("PUT", `/graphs/${id}/versions/active`, {
      active_graph_version: version,
    });
  }

  executeGraph(
    id: GraphID,
    version: number,
    inputs: { [key: string]: any } = {},
    credentials_inputs: { [key: string]: CredentialsMetaInput } = {},
  ): Promise<GraphExecutionMeta> {
    return this._request("POST", `/graphs/${id}/execute/${version}`, {
      inputs,
      credentials_inputs,
    });
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
  ): Promise<{ login_url: string; state_token: string }> {
    const query = scopes ? { scopes: scopes.join(",") } : undefined;
    return this._get(`/integrations/${provider}/login`, query);
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

  listProviders(): Promise<string[]> {
    return this._get("/integrations/providers");
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

  logMetric(metric: AnalyticsMetrics) {
    return this._request("POST", "/analytics/log_raw_metric", metric);
  }

  logAnalytic(analytic: AnalyticsDetails) {
    return this._request("POST", "/analytics/log_raw_analytics", analytic);
  }

  ////////////////////////////////////////
  ///////////// V2 STORE API /////////////
  ////////////////////////////////////////

  getStoreProfile(): Promise<ProfileDetails | null> {
    try {
      const result = this._get("/store/profile");
      return result;
    } catch (error) {
      console.error("Error fetching store profile:", error);
      return Promise.resolve(null);
    }
  }

  getStoreAgents(params?: {
    featured?: boolean;
    creator?: string;
    sorted_by?: string;
    search_query?: string;
    category?: string;
    page?: number;
    page_size?: number;
  }): Promise<StoreAgentsResponse> {
    return this._get("/store/agents", params);
  }

  getStoreAgent(
    username: string,
    agentName: string,
  ): Promise<StoreAgentDetails> {
    return this._get(
      `/store/agents/${encodeURIComponent(username)}/${encodeURIComponent(
        agentName,
      )}`,
    );
  }

  getGraphMetaByStoreListingVersionID(
    storeListingVersionID: string,
  ): Promise<GraphMeta> {
    return this._get(`/store/graph/${storeListingVersionID}`);
  }

  getStoreAgentByVersionId(
    storeListingVersionID: string,
  ): Promise<StoreAgentDetails> {
    return this._get(`/store/agents/${storeListingVersionID}`);
  }

  getStoreCreators(params?: {
    featured?: boolean;
    search_query?: string;
    sorted_by?: string;
    page?: number;
    page_size?: number;
  }): Promise<CreatorsResponse> {
    return this._get("/store/creators", params);
  }

  getStoreCreator(username: string): Promise<CreatorDetails> {
    return this._get(`/store/creator/${encodeURIComponent(username)}`);
  }

  getStoreSubmissions(params?: {
    page?: number;
    page_size?: number;
  }): Promise<StoreSubmissionsResponse> {
    return this._get("/store/submissions", params);
  }

  createStoreSubmission(
    submission: StoreSubmissionRequest,
  ): Promise<StoreSubmission> {
    return this._request("POST", "/store/submissions", submission);
  }

  generateStoreSubmissionImage(
    agent_id: string,
  ): Promise<{ image_url: string }> {
    return this._request(
      "POST",
      "/store/submissions/generate_image?agent_id=" + agent_id,
    );
  }

  deleteStoreSubmission(submission_id: string): Promise<boolean> {
    return this._request("DELETE", `/store/submissions/${submission_id}`);
  }

  uploadStoreSubmissionMedia(file: File): Promise<string> {
    return this._uploadFile("/store/submissions/media", file);
  }

  uploadFile(
    file: File,
    provider: string = "gcs",
    expiration_hours: number = 24,
    onProgress?: (progress: number) => void,
  ): Promise<{
    file_uri: string;
    file_name: string;
    size: number;
    content_type: string;
    expires_in_hours: number;
  }> {
    return this._uploadFileWithProgress(
      "/files/upload",
      file,
      {
        provider,
        expiration_hours,
      },
      onProgress,
    ).then((response) => {
      if (typeof response === "string") {
        return JSON.parse(response);
      }
      return response;
    });
  }

  updateStoreProfile(profile: ProfileDetails): Promise<ProfileDetails> {
    return this._request("POST", "/store/profile", profile);
  }

  reviewAgent(
    username: string,
    agentName: string,
    review: StoreReviewCreate,
  ): Promise<StoreReview> {
    return this._request(
      "POST",
      `/store/agents/${encodeURIComponent(username)}/${encodeURIComponent(
        agentName,
      )}/review`,
      review,
    );
  }

  getMyAgents(params?: {
    page?: number;
    page_size?: number;
  }): Promise<MyAgentsResponse> {
    return this._get("/store/myagents", params);
  }

  downloadStoreAgent(
    storeListingVersionId: string,
    version?: number,
  ): Promise<BlobPart> {
    const url = version
      ? `/store/download/agents/${storeListingVersionId}?version=${version}`
      : `/store/download/agents/${storeListingVersionId}`;

    return this._get(url);
  }

  /////////////////////////////////////////
  /////////// Admin API ///////////////////
  /////////////////////////////////////////

  getAdminListingsWithVersions(params?: {
    status?: SubmissionStatus;
    search?: string;
    page?: number;
    page_size?: number;
  }): Promise<StoreListingsWithVersionsResponse> {
    return this._get("/store/admin/listings", params);
  }

  reviewSubmissionAdmin(
    storeListingVersionId: string,
    review: ReviewSubmissionRequest,
  ): Promise<StoreSubmission> {
    return this._request(
      "POST",
      `/store/admin/submissions/${storeListingVersionId}/review`,
      review,
    );
  }

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

  downloadStoreAgentAdmin(storeListingVersionId: string): Promise<BlobPart> {
    const url = `/store/admin/submissions/download/${storeListingVersionId}`;

    return this._get(url);
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

  addMarketplaceAgentToLibrary(
    storeListingVersionID: string,
  ): Promise<LibraryAgent> {
    return this._request("POST", "/library/agents", {
      store_listing_version_id: storeListingVersionID,
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

  private async getAuthToken(): Promise<string> {
    // Only try client-side session (for WebSocket connections)
    // This will return "no-token-found" with httpOnly cookies, which is expected
    const supabaseClient = await this.getSupabaseClient();
    const {
      data: { session },
    } = (await supabaseClient?.auth.getSession()) || {
      data: { session: null },
    };

    return session?.access_token || "no-token-found";
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
    // has server-only imports (like getServerSupabase) at the top level. Static imports
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
          reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
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
    // has server-only imports (like getServerSupabase) at the top level. Static imports
    // would bundle server-only code into the client bundle, causing runtime errors.
    const { buildClientUrl, buildUrlWithQuery, handleFetchError } =
      await import("./helpers");

    const payloadAsQuery = ["GET", "DELETE"].includes(method);
    let url = buildClientUrl(path);

    if (payloadAsQuery && payload) {
      url = buildUrlWithQuery(url, payload);
    }

    // Prepare headers with admin impersonation support
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (environment.isClientSide()) {
      try {
        const impersonatedUserId = sessionStorage.getItem(
          IMPERSONATION_STORAGE_KEY,
        );
        if (impersonatedUserId) {
          headers[IMPERSONATION_HEADER_NAME] = impersonatedUserId;
        }
      } catch (_error) {
        console.error(
          "Admin impersonation: Failed to access sessionStorage:",
          _error,
        );
      }
    }

    const response = await fetch(url, {
      method,
      headers,
      body: !payloadAsQuery && payload ? JSON.stringify(payload) : undefined,
      credentials: "include",
    });

    if (!response.ok) {
      throw await handleFetchError(response);
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
    return await makeAuthenticatedRequest(method, url, payload);
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
    callCount = 0,
    callCountLimit = 4,
  ): Promise<void> {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify({ method, data }));
      return;
    }
    if (callCount >= callCountLimit) {
      throw new Error(
        `WebSocket connection not open after ${callCountLimit} attempts`,
      );
    }
    await this.connectWebSocket();
    if (callCount === 0) {
      return this.sendWebSocketMessage(method, data, callCount + 1);
    }
    const delayMs = 2 ** (callCount - 1) * 1000;
    await new Promise((res) => setTimeout(res, delayMs));
    return this.sendWebSocketMessage(method, data, callCount + 1);
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

    this.connectWebSocket();
    if (this.webSocket?.readyState == WebSocket.OPEN) handler();

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

  async connectWebSocket(): Promise<void> {
    // Do not attempt to connect if a disconnect intent is present (e.g., during logout)
    if (this._hasDisconnectIntent()) {
      return;
    }

    this.isIntentionallyDisconnected = false;
    return (this.wsConnecting ??= new Promise(async (resolve, reject) => {
      try {
        let token = "";
        try {
          const { token: serverToken, error } = await getWebSocketToken();
          if (serverToken && !error) {
            token = serverToken;
          } else if (error) {
            console.warn("Failed to get WebSocket token from server:", error);
          }
        } catch (error) {
          console.warn("Failed to get token for WebSocket connection:", error);
          // Intentionally fall through; we'll bail out below if no token is available
        }

        // If we don't have a token, skip attempting a connection.
        if (!token) {
          console.info(
            "[BackendAPI] Skipping WebSocket connect: no auth token available",
          );
          // Resolve first, then clear wsConnecting to avoid races for awaiters
          resolve();
          this.wsConnecting = null;
          this.webSocket = null;
          return;
        }

        const wsUrlWithToken = `${this.wsUrl}?token=${token}`;
        this.webSocket = new WebSocket(wsUrlWithToken);
        this.webSocket.state = "connecting";

        this.webSocket.onopen = () => {
          this.webSocket!.state = "connected";
          console.info("[BackendAPI] WebSocket connected to", this.wsUrl);
          this._startWSHeartbeat(); // Start heartbeat when connection opens
          this._clearDisconnectIntent(); // Clear disconnect intent when connected
          this.wsOnConnectHandlers.forEach((handler) => handler());
          resolve();
        };

        this.webSocket.onclose = (event) => {
          if (this.webSocket?.state == "connecting") {
            console.error(
              `[BackendAPI] WebSocket failed to connect: ${event.reason}`,
              event,
            );
          } else if (this.webSocket?.state == "connected") {
            console.warn(
              `[BackendAPI] WebSocket connection closed: ${event.reason}`,
              event,
            );
          }
          this.webSocket!.state = "closed";

          this._stopWSHeartbeat(); // Stop heartbeat when connection closes
          this.wsConnecting = null;

          const wasIntentional =
            this.isIntentionallyDisconnected || this._hasDisconnectIntent();

          if (!wasIntentional) {
            this.wsOnDisconnectHandlers.forEach((handler) => handler());
            setTimeout(() => this.connectWebSocket().then(resolve), 1000);
          } else {
            // Ensure pending connect calls settle on intentional close
            resolve();
          }
        };

        this.webSocket.onerror = (error) => {
          if (this.webSocket?.state == "connected") {
            console.error("[BackendAPI] WebSocket error:", error);
          }
        };
        this.webSocket.onmessage = (event) => this._handleWSMessage(event);
      } catch (error) {
        console.error("[BackendAPI] Error connecting to WebSocket:", error);
        reject(error);
      }
    }));
  }

  disconnectWebSocket() {
    this.isIntentionallyDisconnected = true;
    this._stopWSHeartbeat(); // Stop heartbeat when disconnecting
    if (
      this.webSocket &&
      (this.webSocket.readyState === WebSocket.OPEN ||
        this.webSocket.readyState === WebSocket.CONNECTING)
    ) {
      this.webSocket.close();
    }
    this.wsConnecting = null;
  }

  private _hasDisconnectIntent(): boolean {
    if (!isClient) return false;

    try {
      return storage.get(Key.WEBSOCKET_DISCONNECT_INTENT) === "true";
    } catch {
      return false;
    }
  }

  private _clearDisconnectIntent(): void {
    if (!isClient) return;

    try {
      storage.clean(Key.WEBSOCKET_DISCONNECT_INTENT);
    } catch {
      Sentry.captureException(
        new Error("Failed to clear WebSocket disconnect intent"),
      );
    }
  }

  private _handleWSMessage(event: MessageEvent): void {
    const message: WebsocketMessage = JSON.parse(event.data);

    // Handle heartbeat response
    if (message.method === "heartbeat" && message.data === "pong") {
      this._handleWSHeartbeatResponse();
      return;
    }

    if (message.method === "node_execution_event") {
      message.data = parseNodeExecutionResultTimestamps(message.data);
    } else if (message.method == "graph_execution_event") {
      message.data = parseGraphExecutionTimestamps(message.data);
    }
    this.wsMessageHandlers[message.method]?.forEach((handler) =>
      handler(message.data),
    );
  }

  private _startWSHeartbeat() {
    this._stopWSHeartbeat();
    this.heartbeatIntervalID = window.setInterval(() => {
      if (this.webSocket?.readyState === WebSocket.OPEN) {
        this.webSocket.send(
          JSON.stringify({
            method: "heartbeat",
            data: "ping",
            success: true,
          }),
        );

        this.heartbeatTimeoutID = window.setTimeout(() => {
          console.warn("Heartbeat timeout - reconnecting");
          this.webSocket?.close();
          this.connectWebSocket();
        }, this.HEARTBEAT_TIMEOUT);
      }
    }, this.HEARTBEAT_INTERVAL);
  }

  private _stopWSHeartbeat() {
    if (this.heartbeatIntervalID) {
      clearInterval(this.heartbeatIntervalID);
      this.heartbeatIntervalID = null;
    }
    if (this.heartbeatTimeoutID) {
      clearTimeout(this.heartbeatTimeoutID);
      this.heartbeatTimeoutID = null;
    }
  }

  private _handleWSHeartbeatResponse() {
    if (this.heartbeatTimeoutID) {
      clearTimeout(this.heartbeatTimeoutID);
      this.heartbeatTimeoutID = null;
    }
  }
}

declare global {
  interface WebSocket {
    state: "connecting" | "connected" | "closed";
  }
}

/* *** UTILITY TYPES *** */

type GraphCreateRequestBody = {
  graph: GraphCreatable;
};

type WebsocketMessageTypeMap = {
  subscribe_graph_execution: { graph_exec_id: GraphExecutionID };
  subscribe_graph_executions: { graph_id: GraphID };
  graph_execution_event: GraphExecution;
  node_execution_event: NodeExecutionResult;
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
