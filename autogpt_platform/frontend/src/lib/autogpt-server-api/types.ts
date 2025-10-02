export enum SubmissionStatus {
  DRAFT = "DRAFT",
  PENDING = "PENDING",
  APPROVED = "APPROVED",
  REJECTED = "REJECTED",
}
export type ReviewSubmissionRequest = {
  store_listing_version_id: string;
  is_approved: boolean;
  comments: string; // External comments visible to creator
  internal_comments?: string; // Admin-only comments
};
export type Category = {
  category: string;
  description: string;
};

export enum BlockCostType {
  RUN = "run",
  BYTE = "byte",
  SECOND = "second",
}

export type BlockCost = {
  cost_amount: number;
  cost_type: BlockCostType;
  cost_filter: Record<string, any>;
};

/* Mirror of backend/data/block.py:Block */
export type Block = {
  id: string;
  name: string;
  description: string;
  categories: Category[];
  inputSchema: BlockIORootSchema;
  outputSchema: BlockIORootSchema;
  staticOutput: boolean;
  uiType: BlockUIType;
  costs: BlockCost[];
};

export type BlockIORootSchema = {
  type: "object";
  properties: Record<string, BlockIOSubSchema>;
  required?: (keyof BlockIORootSchema["properties"])[];
  additionalProperties?: { type: string };
};

export type BlockIOSubSchema =
  | BlockIOSimpleTypeSubSchema
  | BlockIOCombinedTypeSubSchema;

export type BlockIOSubType = BlockIOSimpleTypeSubSchema["type"];

export type BlockIOSimpleTypeSubSchema =
  | BlockIOObjectSubSchema
  | BlockIOCredentialsSubSchema
  | BlockIOKVSubSchema
  | BlockIOArraySubSchema
  | BlockIOTableSubSchema
  | BlockIOStringSubSchema
  | BlockIONumberSubSchema
  | BlockIOBooleanSubSchema
  | BlockIONullSubSchema;

export enum DataType {
  SHORT_TEXT = "short-text",
  LONG_TEXT = "long-text",
  NUMBER = "number",
  DATE = "date",
  TIME = "time",
  DATE_TIME = "date-time",
  FILE = "file",
  SELECT = "select",
  MULTI_SELECT = "multi-select",
  BOOLEAN = "boolean",
  CREDENTIALS = "credentials",
  OBJECT = "object",
  KEY_VALUE = "key-value",
  ARRAY = "array",
  TABLE = "table",
}

export type BlockIOSubSchemaMeta = {
  title?: string;
  description?: string;
  placeholder?: string;
  advanced?: boolean;
  depends_on?: string[];
  hidden?: boolean;
};

export type BlockIOObjectSubSchema = BlockIOSubSchemaMeta & {
  type: "object";
  properties: Record<string, BlockIOSubSchema>;
  const?: Record<keyof BlockIOObjectSubSchema["properties"], any>;
  default?: Record<keyof BlockIOObjectSubSchema["properties"], any>;
  required?: (keyof BlockIOObjectSubSchema["properties"])[];
  secret?: boolean;
};

export type BlockIOKVSubSchema = BlockIOSubSchemaMeta & {
  type: "object";
  additionalProperties?: { type: "string" | "number" | "integer" };
  const?: Record<string, string | number>;
  default?: Record<string, string | number>;
  secret?: boolean;
};

export type BlockIOArraySubSchema = BlockIOSubSchemaMeta & {
  type: "array";
  items?: BlockIOSimpleTypeSubSchema;
  const?: Array<string>;
  default?: Array<string>;
  secret?: boolean;
};

// Table cell values are typically primitives
export type TableCellValue = string | number | boolean | null;

export type TableRow = Record<string, TableCellValue>;

export type BlockIOTableSubSchema = BlockIOSubSchemaMeta & {
  type: "array";
  format: "table";
  items: BlockIOObjectSubSchema;
  const?: TableRow[];
  default?: TableRow[];
  secret?: boolean;
};

export type BlockIOStringSubSchema = BlockIOSubSchemaMeta & {
  type: "string";
  enum?: string[];
  secret?: true;
  const?: string;
  default?: string;
  format?: string;
  maxLength?: number;
};

export type BlockIONumberSubSchema = BlockIOSubSchemaMeta & {
  type: "integer" | "number";
  const?: number;
  default?: number;
  secret?: boolean;
};

export type BlockIOBooleanSubSchema = BlockIOSubSchemaMeta & {
  type: "boolean";
  const?: boolean;
  default?: boolean;
  secret?: boolean;
};

export type CredentialsType =
  | "api_key"
  | "oauth2"
  | "user_password"
  | "host_scoped";

export type Credentials =
  | APIKeyCredentials
  | OAuth2Credentials
  | UserPasswordCredentials
  | HostScopedCredentials;

// --8<-- [start:BlockIOCredentialsSubSchema]
// Provider names are now dynamic and fetched from the API
// This allows for SDK-registered providers without hardcoding
export type CredentialsProviderName = string;

// For backward compatibility, we'll keep PROVIDER_NAMES but it should be
// populated dynamically from the API. This is a placeholder that will be
// replaced with actual values from the /api/integrations/providers endpoint
export const PROVIDER_NAMES = {} as Record<string, string>;
// --8<-- [end:BlockIOCredentialsSubSchema]

export type BlockIOCredentialsSubSchema = BlockIOObjectSubSchema & {
  /* Mirror of backend/data/model.py:CredentialsFieldSchemaExtra */
  credentials_provider: CredentialsProviderName[];
  credentials_scopes?: string[];
  credentials_types: Array<CredentialsType>;
  discriminator?: string;
  discriminator_mapping?: Record<string, CredentialsProviderName>;
  discriminator_values?: any[];
  secret?: boolean;
};

export type BlockIONullSubSchema = BlockIOSubSchemaMeta & {
  type: "null";
  const?: null;
  secret?: boolean;
};

// At the time of writing, combined schemas only occur on the first nested level in a
// block schema. It is typed this way to make the use of these objects less tedious.
type BlockIOCombinedTypeSubSchema = BlockIOSubSchemaMeta & {
  type: never;
  const: never;
} & (
    | {
        allOf: [BlockIOSimpleTypeSubSchema];
        secret?: boolean;
      }
    | {
        anyOf: BlockIOSimpleTypeSubSchema[];
        default?: string | number | boolean | null;
        secret?: boolean;
        format?: string; // For table format and other formats on anyOf schemas
      }
    | BlockIOOneOfSubSchema
    | BlockIODiscriminatedOneOfSubSchema
  );

export type BlockIOOneOfSubSchema = {
  oneOf: BlockIOSimpleTypeSubSchema[];
  default?: string | number | boolean | null;
  secret?: boolean;
};

export type BlockIODiscriminatedOneOfSubSchema = {
  oneOf: BlockIOObjectSubSchema[];
  discriminator: {
    propertyName: string;
    mapping: Record<string, BlockIOObjectSubSchema>;
  };
  default?: Record<string, any>;
  secret?: boolean;
};

export type NodeCreatable = {
  id: string;
  block_id: string;
  input_default: Record<string, any>;
  metadata: {
    position: { x: number; y: number };
    [key: string]: any;
  };
};

/* Mirror of backend/data/graph.py:Node */
export type Node = NodeCreatable & {
  input_links: Link[];
  output_links: Link[];
  webhook?: Webhook;
};

/* Mirror of backend/data/graph.py:Link */
export type Link = {
  id: string;
  source_id: string;
  sink_id: string;
  source_name: string;
  sink_name: string;
  is_static: boolean;
};

export type LinkCreatable = Omit<Link, "id" | "is_static"> & {
  id?: string;
};

/* Mirror of backend/data/execution.py:GraphExecutionMeta */
export type GraphExecutionMeta = {
  id: GraphExecutionID;
  user_id: UserID;
  graph_id: GraphID;
  graph_version: number;
  inputs: Record<string, any> | null;
  credential_inputs: Record<string, CredentialsMetaInput> | null;
  nodes_input_masks: Record<string, Record<string, any>> | null;
  preset_id: LibraryAgentPresetID | null;
  status:
    | "QUEUED"
    | "RUNNING"
    | "COMPLETED"
    | "TERMINATED"
    | "FAILED"
    | "INCOMPLETE";
  started_at: Date;
  ended_at: Date;
  stats: {
    error: string | null;
    cost: number;
    duration: number;
    duration_cpu_only: number;
    node_exec_time: number;
    node_exec_time_cpu_only: number;
    node_exec_count: number;
    activity_status: string | null;
    [key: string]: any;
  } | null;
};

export type GraphExecutionID = Brand<string, "GraphExecutionID">;

/* Mirror of backend/data/execution.py:GraphExecution */
export type GraphExecution = Omit<GraphExecutionMeta, "inputs"> & {
  inputs: Record<string, any>;
  outputs: Record<string, Array<any>>;
  node_executions?: NodeExecutionResult[];
};

export type GraphExecutionsResponse = {
  executions: GraphExecutionMeta[];
  pagination: Pagination;
};

/* Mirror of backend/data/graph.py:GraphMeta */
export type GraphMeta = {
  id: GraphID;
  user_id: UserID;
  version: number;
  is_active: boolean;
  name: string;
  description: string;
  instructions?: string | null;
  recommended_schedule_cron: string | null;
  forked_from_id?: GraphID | null;
  forked_from_version?: number | null;
  input_schema: GraphIOSchema;
  output_schema: GraphIOSchema;
  credentials_input_schema: CredentialsInputSchema;
} & (
  | {
      has_external_trigger: true;
      trigger_setup_info: GraphTriggerInfo;
    }
  | {
      has_external_trigger: false;
      trigger_setup_info: null;
    }
);

export type GraphID = Brand<string, "GraphID">;

/* Derived from backend/data/graph.py:Graph._generate_schema() */
export type GraphIOSchema = {
  type: "object";
  properties: Record<string, GraphIOSubSchema>;
  required: (keyof BlockIORootSchema["properties"])[];
};
export type GraphIOSubSchema = Omit<
  BlockIOSubSchemaMeta,
  "placeholder" | "depends_on" | "hidden"
> & {
  type: never; // bodge to avoid type checking hell; doesn't exist at runtime
  default?: string;
  secret: boolean;
  metadata?: any;
};

export type CredentialsInputSchema = {
  type: "object";
  properties: Record<string, BlockIOCredentialsSubSchema>;
  required?: (keyof CredentialsInputSchema["properties"])[];
};

/* Mirror of backend/data/graph.py:GraphTriggerInfo */
export type GraphTriggerInfo = {
  provider: CredentialsProviderName;
  config_schema: BlockIORootSchema;
  credentials_input_name: string | null;
};

/* Mirror of backend/data/graph.py:Graph */
export type Graph = GraphMeta & {
  created_at: Date;
  nodes: Node[];
  links: Link[];
  sub_graphs: Omit<Graph, "sub_graphs">[]; // Flattened sub-graphs
};

export type GraphUpdateable = Omit<
  Graph,
  | "user_id"
  | "version"
  | "created_at"
  | "is_active"
  | "nodes"
  | "links"
  | "sub_graphs"
  | "input_schema"
  | "output_schema"
  | "credentials_input_schema"
  | "has_external_trigger"
  | "trigger_setup_info"
> & {
  version?: number;
  is_active?: boolean;
  nodes: NodeCreatable[];
  links: LinkCreatable[];
  input_schema?: GraphIOSchema;
  output_schema?: GraphIOSchema;
};

export type GraphCreatable = _GraphCreatableInner & {
  sub_graphs?: _GraphCreatableInner[]; // Flattened sub-graphs
};
type _GraphCreatableInner = Omit<GraphUpdateable, "id"> & { id?: string };

/* Mirror of backend/data/execution.py:NodeExecutionResult */
export type NodeExecutionResult = {
  graph_id: GraphID;
  graph_version: number;
  graph_exec_id: GraphExecutionID;
  node_exec_id: string;
  node_id: string;
  block_id: string;
  status:
    | "INCOMPLETE"
    | "QUEUED"
    | "RUNNING"
    | "COMPLETED"
    | "TERMINATED"
    | "FAILED";
  input_data: Record<string, any>;
  output_data: Record<string, Array<any>>;
  add_time: Date;
  queue_time?: Date;
  start_time?: Date;
  end_time?: Date;
};

/* Structured validation error types for graph execution */
export type GraphValidationErrorResponse = {
  detail: {
    type: "validation_error";
    message: string;
    node_errors: Record<string, Record<string, string>>;
  };
};

/* *** LIBRARY *** */

/* Mirror of backend/server/v2/library/model.py:LibraryAgent */
export type LibraryAgent = {
  id: LibraryAgentID;
  graph_id: GraphID;
  graph_version: number;
  image_url: string | null;
  creator_name: string;
  creator_image_url: string;
  status: AgentStatus;
  updated_at: Date;
  name: string;
  description: string;
  instructions?: string | null;
  input_schema: GraphIOSchema;
  output_schema: GraphIOSchema;
  credentials_input_schema: CredentialsInputSchema;
  new_output: boolean;
  can_access_graph: boolean;
  is_favorite: boolean;
  is_latest_version: boolean;
  recommended_schedule_cron: string | null;
} & (
  | {
      has_external_trigger: true;
      trigger_setup_info: GraphTriggerInfo;
    }
  | {
      has_external_trigger: false;
      trigger_setup_info: null;
    }
);

export type LibraryAgentID = Brand<string, "LibraryAgentID">;

export enum AgentStatus {
  COMPLETED = "COMPLETED",
  HEALTHY = "HEALTHY",
  WAITING = "WAITING",
  ERROR = "ERROR",
}

export type LibraryAgentResponse = {
  agents: LibraryAgent[];
  pagination: Pagination;
};

export type LibraryAgentPreset = {
  id: LibraryAgentPresetID;
  created_at: Date;
  updated_at: Date;
  graph_id: GraphID;
  graph_version: number;
  inputs: Record<string, any>;
  credentials: Record<string, CredentialsMetaInput>;
  name: string;
  description: string;
  is_active: boolean;
} & (
  | {
      webhook_id: string;
      webhook: Webhook;
    }
  | {
      webhook_id?: undefined;
      webhook?: undefined;
    }
);

export type LibraryAgentPresetID = Brand<string, "LibraryAgentPresetID">;

export type LibraryAgentPresetResponse = {
  presets: LibraryAgentPreset[];
  pagination: Pagination;
};

export type LibraryAgentPresetCreatable = Omit<
  LibraryAgentPreset,
  "id" | "created_at" | "updated_at" | "is_active"
> & {
  is_active?: boolean;
};

export type LibraryAgentPresetCreatableFromGraphExecution = Omit<
  LibraryAgentPresetCreatable,
  "graph_id" | "graph_version" | "inputs" | "credentials"
> & {
  graph_execution_id: GraphExecutionID;
};

export type LibraryAgentPresetUpdatable = Partial<
  Omit<LibraryAgentPresetCreatable, "graph_id" | "graph_version">
>;

export enum LibraryAgentSortEnum {
  CREATED_AT = "createdAt",
  UPDATED_AT = "updatedAt",
}

/* *** CREDENTIALS *** */

/* Mirror of backend/server/integrations/router.py:CredentialsMetaResponse */
export type CredentialsMetaResponse = {
  id: string;
  provider: CredentialsProviderName;
  type: CredentialsType;
  title?: string;
  scopes?: Array<string>;
  username?: string;
  host?: string;
};

/* Mirror of backend/server/integrations/router.py:CredentialsDeletionResponse */
export type CredentialsDeleteResponse = {
  deleted: true;
  revoked: boolean | null;
};

/* Mirror of backend/server/integrations/router.py:CredentialsDeletionNeedsConfirmationResponse */
export type CredentialsDeleteNeedConfirmationResponse = {
  deleted: false;
  need_confirmation: true;
  message: string;
};

/* Mirror of backend/data/model.py:CredentialsMetaInput */
export type CredentialsMetaInput = {
  id: string;
  type: CredentialsType;
  title?: string | null;
  provider: string;
};

/* Mirror of backend/backend/data/model.py:_BaseCredentials */
type BaseCredentials = {
  id: string;
  type: CredentialsType;
  title?: string;
  provider: CredentialsProviderName;
};

/* Mirror of backend/backend/data/model.py:OAuth2Credentials */
export type OAuth2Credentials = BaseCredentials & {
  type: "oauth2";
  scopes: string[];
  username?: string;
  access_token: string;
  access_token_expires_at?: number;
  refresh_token?: string;
  refresh_token_expires_at?: number;
  metadata: Record<string, any>;
};

/* Mirror of backend/backend/data/model.py:APIKeyCredentials */
export type APIKeyCredentials = BaseCredentials & {
  type: "api_key";
  title: string;
  api_key: string;
  expires_at?: number;
};

export type UserPasswordCredentials = BaseCredentials & {
  type: "user_password";
  title: string;
  username: string;
  password: string;
};

/* Mirror of backend/backend/data/model.py:HostScopedCredentials */
export type HostScopedCredentials = BaseCredentials & {
  type: "host_scoped";
  title: string;
  host: string;
  headers: Record<string, string>;
};

// Mirror of backend/backend/data/notifications.py:NotificationType
export type NotificationType =
  | "AGENT_RUN"
  | "ZERO_BALANCE"
  | "LOW_BALANCE"
  | "BLOCK_EXECUTION_FAILED"
  | "CONTINUOUS_AGENT_ERROR"
  | "DAILY_SUMMARY"
  | "WEEKLY_SUMMARY"
  | "MONTHLY_SUMMARY"
  | "AGENT_APPROVED"
  | "AGENT_REJECTED";

// Mirror of backend/backend/data/notifications.py:NotificationPreference
export type NotificationPreferenceDTO = {
  email: string;
  preferences: { [key in NotificationType]: boolean };
  daily_limit: number;
};

export type NotificationPreference = NotificationPreferenceDTO & {
  user_id: UserID;
  emails_sent_today: number;
  last_reset_date: Date;
};

/* Mirror of backend/data/integrations.py:Webhook */
export type Webhook = {
  id: string;
  url: string;
  provider: CredentialsProviderName;
  credentials_id: string; // empty string if not appicable
  webhook_type: string;
  resource: string; // empty string if not appicable
  events: string[];
  secret: string;
  config: Record<string, any>;
  provider_webhook_id?: string;
};

export type User = {
  id: UserID;
  email: string;
};

export type UserID = Brand<string, "UserID">;

export enum BlockUIType {
  STANDARD = "Standard",
  INPUT = "Input",
  OUTPUT = "Output",
  NOTE = "Note",
  WEBHOOK = "Webhook",
  WEBHOOK_MANUAL = "Webhook (manual)",
  AGENT = "Agent",
  AI = "AI",
  AYRSHARE = "Ayrshare",
}

export enum SpecialBlockID {
  AGENT = "e189baac-8c20-45a1-94a7-55177ea42565",
  SMART_DECISION = "3b191d9f-356f-482d-8238-ba04b6d18381",
  OUTPUT = "363ae599-353e-4804-937e-b2ee3cef3da4",
}

export type AnalyticsMetrics = {
  metric_name: string;
  metric_value: number;
  data_string: string;
};

export type AnalyticsDetails = {
  type: string;
  data: Record<string, any>;
  index: string;
};

export type Pagination = {
  total_items: number;
  total_pages: number;
  current_page: number;
  page_size: number;
};

export type StoreAgent = {
  slug: string;
  agent_name: string;
  agent_image: string;
  creator: string;
  creator_avatar: string;
  sub_heading: string;
  description: string;
  runs: number;
  rating: number;
  updated_at: string;
};

export type StoreAgentsResponse = {
  agents: StoreAgent[];
  pagination: Pagination;
};

export type StoreAgentDetails = {
  store_listing_version_id: string;
  slug: string;
  updated_at: string;
  agent_name: string;
  agent_video: string;
  agent_image: string[];
  creator: string;
  creator_avatar: string;
  sub_heading: string;
  description: string;
  categories: string[];
  runs: number;
  rating: number;
  versions: string[];

  // Approval and status fields
  active_version_id?: string;
  has_approved_version?: boolean;
  is_available?: boolean;
};

export type Creator = {
  name: string;
  username: string;
  description: string;
  avatar_url: string;
  num_agents: number;
  agent_rating: number;
  agent_runs: number;
};

export type CreatorsResponse = {
  creators: Creator[];
  pagination: Pagination;
};

export type CreatorDetails = {
  name: string;
  username: string;
  description: string;
  links: string[];
  avatar_url: string;
  agent_rating: number;
  agent_runs: number;
  top_categories: string[];
};

export type StoreSubmission = {
  agent_id: string;
  agent_version: number;
  name: string;
  sub_heading: string;
  description: string;
  instructions?: string;
  image_urls: string[];
  date_submitted: string;
  status: SubmissionStatus;
  runs: number;
  rating: number;
  slug: string;
  store_listing_version_id?: string;
  version?: number; // Actual version number from the database

  // Review information
  reviewer_id?: string;
  review_comments?: string;
  internal_comments?: string; // Admin-only comments
  reviewed_at?: string;
  changes_summary?: string;
};

export type StoreSubmissionsResponse = {
  submissions: StoreSubmission[];
  pagination: Pagination;
};

export type StoreSubmissionRequest = {
  agent_id: string;
  agent_version: number;
  slug: string;
  name: string;
  sub_heading: string;
  video_url?: string;
  image_urls: string[];
  description: string;
  instructions?: string | null;
  categories: string[];
  changes_summary?: string;
  recommended_schedule_cron?: string | null;
};

export type ProfileDetails = {
  name: string;
  username: string;
  description: string;
  links: string[];
  avatar_url: string;
};

/* Mirror of backend/executor/scheduler.py:GraphExecutionJobInfo */
export type Schedule = {
  id: ScheduleID;
  name: string;
  cron: string;
  user_id: UserID;
  graph_id: GraphID;
  graph_version: number;
  input_data: Record<string, any>;
  input_credentials: Record<string, CredentialsMetaInput>;
  next_run_time: Date;
  timezone: string;
};

export type ScheduleID = Brand<string, "ScheduleID">;

/* Mirror of backend/server/routers/v1.py:ScheduleCreationRequest */
export type ScheduleCreatable = {
  graph_id: GraphID;
  graph_version: number;
  name: string;
  cron: string;
  inputs: Record<string, any>;
  credentials?: Record<string, CredentialsMetaInput>;
};

export type MyAgent = {
  agent_id: GraphID;
  agent_version: number;
  agent_name: string;
  agent_image: string | null;
  last_edited: string;
  description: string;
  recommended_schedule_cron: string | null;
};

export type MyAgentsResponse = {
  agents: MyAgent[];
  pagination: Pagination;
};

export type StoreReview = {
  score: number;
  comments?: string;
};

export type StoreReviewCreate = {
  store_listing_version_id: string;
  score: number;
  comments?: string;
};

// API Key Types

export enum APIKeyPermission {
  EXECUTE_GRAPH = "EXECUTE_GRAPH",
  READ_GRAPH = "READ_GRAPH",
  EXECUTE_BLOCK = "EXECUTE_BLOCK",
  READ_BLOCK = "READ_BLOCK",
}

export enum APIKeyStatus {
  ACTIVE = "ACTIVE",
  REVOKED = "REVOKED",
  SUSPENDED = "SUSPENDED",
}

export interface APIKey {
  id: string;
  name: string;
  prefix: string;
  postfix: string;
  status: APIKeyStatus;
  permissions: APIKeyPermission[];
  created_at: string;
  last_used_at?: string;
  revoked_at?: string;
  description?: string;
}

export interface CreateAPIKeyResponse {
  api_key: APIKey;
  plain_text_key: string;
}

export interface CreditTransaction {
  transaction_key: string;
  transaction_time: Date;
  transaction_type: CreditTransactionType;
  amount: number;
  running_balance: number;
  current_balance: number;
  description: string;
  usage_graph_id: GraphID;
  usage_execution_id: GraphExecutionID;
  usage_node_count: number;
  usage_starting_time: Date;
  user_id: string;
  user_email: string | null;
  reason: string | null;
  admin_email: string | null;
  extra_data: string | null;
}

export interface TransactionHistory {
  transactions: CreditTransaction[];
  next_transaction_time: Date | null;
}

export interface RefundRequest {
  id: string;
  user_id: UserID;
  transaction_key: string;
  amount: number;
  reason: string;
  result: string | null;
  status: string;
  created_at: Date;
  updated_at: Date;
}

export type OnboardingStep =
  // Introductory onboarding (Library)
  | "WELCOME"
  | "USAGE_REASON"
  | "INTEGRATIONS"
  | "AGENT_CHOICE"
  | "AGENT_NEW_RUN"
  | "AGENT_INPUT"
  | "CONGRATS"
  // First Wins
  | "GET_RESULTS"
  | "MARKETPLACE_VISIT"
  | "MARKETPLACE_ADD_AGENT"
  | "MARKETPLACE_RUN_AGENT"
  | "BUILDER_SAVE_AGENT"
  // Consistency Challenge
  | "RE_RUN_AGENT"
  | "SCHEDULE_AGENT"
  | "RUN_AGENTS"
  | "RUN_3_DAYS"
  // The Pro Playground
  | "TRIGGER_WEBHOOK"
  | "RUN_14_DAYS"
  | "RUN_AGENTS_100"
  // No longer used but tracked
  | "BUILDER_OPEN"
  | "BUILDER_RUN_AGENT";

export interface UserOnboarding {
  completedSteps: OnboardingStep[];
  walletShown: boolean;
  notified: OnboardingStep[];
  rewardedFor: OnboardingStep[];
  usageReason: string | null;
  integrations: string[];
  otherIntegrations: string | null;
  selectedStoreListingVersionId: string | null;
  agentInput: Record<string, string | number> | null;
  onboardingAgentExecutionId: GraphExecutionID | null;
  lastRunAt: Date | null;
  consecutiveRunDays: number;
  agentRuns: number;
}

/* *** UTILITIES *** */

/** Use branded types for IDs -> deny mixing IDs between different object classes */
export type Brand<T, Brand extends string> = T & {
  readonly [B in Brand as `__${B}_brand`]: never;
};

export interface OttoDocument {
  url: string;
  relevance_score: number;
}

export interface OttoResponse {
  answer: string;
  documents: OttoDocument[];
  success: boolean;
  error: boolean;
}

export interface OttoQuery {
  query: string;
  conversation_history: { query: string; response: string }[];
  message_id: string;
  include_graph_data: boolean;
  graph_id?: string;
}

export interface StoreListingWithVersions {
  listing_id: string;
  slug: string;
  agent_id: string;
  agent_version: number;
  active_version_id: string | null;
  has_approved_version: boolean;
  creator_email: string | null;
  latest_version: StoreSubmission | null;
  versions: StoreSubmission[];
}

export interface StoreListingsWithVersionsResponse {
  listings: StoreListingWithVersions[];
  pagination: Pagination;
}

// Admin API Types
export type AdminSubmissionsRequest = {
  status?: SubmissionStatus;
  search?: string;
  page: number;
  page_size: number;
};

export type AdminListingHistoryRequest = {
  listing_id: string;
  page: number;
  page_size: number;
};

export type AdminSubmissionDetailsRequest = {
  store_listing_version_id: string;
};

export type AdminPendingSubmissionsRequest = {
  page: number;
  page_size: number;
};

export enum CreditTransactionType {
  TOP_UP = "TOP_UP",
  USAGE = "USAGE",
  GRANT = "GRANT",
  REFUND = "REFUND",
  CARD_CHECK = "CARD_CHECK",
}

export type UsersBalanceHistoryResponse = {
  history: CreditTransaction[];
  pagination: Pagination;
};

export type AddUserCreditsResponse = {
  new_balance: number;
  transaction_key: string;
};
const _stringFormatToDataTypeMap: Partial<Record<string, DataType>> = {
  date: DataType.DATE,
  time: DataType.TIME,
  file: DataType.FILE,
  "date-time": DataType.DATE_TIME,
  "short-text": DataType.SHORT_TEXT,
  "long-text": DataType.LONG_TEXT,
};

function _handleStringSchema(strSchema: BlockIOStringSubSchema): DataType {
  if (strSchema.format) {
    const type = _stringFormatToDataTypeMap[strSchema.format];
    if (type) return type;
  }
  if (strSchema.enum) return DataType.SELECT;
  if (strSchema.maxLength && strSchema.maxLength > 200)
    return DataType.LONG_TEXT;
  return DataType.SHORT_TEXT;
}

function _handleSingleTypeSchema(subSchema: BlockIOSubSchema): DataType {
  if (subSchema.type === "string") {
    return _handleStringSchema(subSchema as BlockIOStringSubSchema);
  }
  if (subSchema.type === "boolean") {
    return DataType.BOOLEAN;
  }
  if (subSchema.type === "number" || subSchema.type === "integer") {
    return DataType.NUMBER;
  }
  if (subSchema.type === "array") {
    // Check for table format first
    if ("format" in subSchema && subSchema.format === "table") {
      return DataType.TABLE;
    }
    /** Commented code below since we haven't yet support rendering of a multi-select with array { items: enum } type */
    // if ("items" in subSchema && subSchema.items && "enum" in subSchema.items) {
    //   return DataType.MULTI_SELECT; // array + enum => multi-select
    // }
    return DataType.ARRAY;
  }
  if (subSchema.type === "object") {
    if (
      ("additionalProperties" in subSchema && subSchema.additionalProperties) ||
      !("properties" in subSchema)
    ) {
      return DataType.KEY_VALUE; // if additionalProperties / no properties => key-value
    }
    if (
      Object.values(subSchema.properties).every(
        (prop) => prop.type === "boolean",
      )
    ) {
      return DataType.MULTI_SELECT; // if all props are boolean => multi-select
    }
    return DataType.OBJECT;
  }
  return DataType.SHORT_TEXT;
}

export function determineDataType(schema: BlockIOSubSchema): DataType {
  if ("allOf" in schema) {
    // If this happens, that is because Pydantic wraps $refs in an allOf if the
    // $ref has sibling schema properties (which isn't technically allowed),
    // so there will only be one item in allOf[].
    // this should NEVER happen though, as $refs are resolved server-side.
    console.warn(
      `Detected 'allOf' wrapper: ${schema}. Normalizing use ${schema.allOf[0]} instead.`,
    );
    schema = schema.allOf[0];
  }

  // Credentials override
  if ("credentials_provider" in schema) {
    return DataType.CREDENTIALS;
  }

  // enum == SELECT
  if ("enum" in schema) {
    return DataType.SELECT;
  }

  // Handle anyOf => optional types (string|null, number|null, etc.)
  if ("anyOf" in schema) {
    // e.g. schema.anyOf might look like [{ type: "string", ... }, { type: "null" }]
    const types = schema.anyOf.map((sub) =>
      "type" in sub ? sub.type : undefined,
    );

    // (string | null)
    if (types.includes("string") && types.includes("null")) {
      const strSchema = schema.anyOf.find(
        (s) => s.type === "string",
      ) as BlockIOStringSubSchema;
      return _handleStringSchema(strSchema);
    }

    // (number|integer) & null
    if (
      (types.includes("number") || types.includes("integer")) &&
      types.includes("null")
    ) {
      // Just reuse our single-type logic for whichever is not null
      const numSchema = schema.anyOf.find(
        (s) => s.type === "number" || s.type === "integer",
      );
      if (numSchema) {
        return _handleSingleTypeSchema(numSchema);
      }
      return DataType.NUMBER; // fallback
    }

    // (array | null)
    if (types.includes("array") && types.includes("null")) {
      // Check for table format on the parent schema (where anyOf is)
      if ("format" in schema && schema.format === "table") {
        return DataType.TABLE;
      }

      const arrSchema = schema.anyOf.find((s) => s.type === "array");
      if (arrSchema) return _handleSingleTypeSchema(arrSchema);
      return DataType.ARRAY;
    }

    // (object | null)
    if (types.includes("object") && types.includes("null")) {
      const objSchema = schema.anyOf.find(
        (s) => s.type === "object",
      ) as BlockIOObjectSubSchema;
      if (objSchema) return _handleSingleTypeSchema(objSchema);
      return DataType.OBJECT;
    }
  }

  // oneOf + discriminator => user picks which variant => SELECT
  if ("oneOf" in schema && "discriminator" in schema && schema.discriminator) {
    return DataType.SELECT;
  }

  // Direct type
  if ("type" in schema) {
    return _handleSingleTypeSchema(schema);
  }

  // Fallback
  return DataType.SHORT_TEXT;
}
