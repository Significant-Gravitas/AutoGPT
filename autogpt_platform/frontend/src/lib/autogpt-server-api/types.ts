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
  cost_filter: { [key: string]: any };
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
  uiKey?: string;
  costs: BlockCost[];
  hardcodedValues: { [key: string]: any } | null;
};

export type BlockIORootSchema = {
  type: "object";
  properties: { [key: string]: BlockIOSubSchema };
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
  | BlockIOStringSubSchema
  | BlockIONumberSubSchema
  | BlockIOBooleanSubSchema
  | BlockIONullSubSchema;

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
  properties: { [key: string]: BlockIOSubSchema };
  default?: { [key: keyof BlockIOObjectSubSchema["properties"]]: any };
  required?: (keyof BlockIOObjectSubSchema["properties"])[];
  secret?: boolean;
};

export type BlockIOKVSubSchema = BlockIOSubSchemaMeta & {
  type: "object";
  additionalProperties?: { type: "string" | "number" | "integer" };
  default?: { [key: string]: string | number };
  secret?: boolean;
};

export type BlockIOArraySubSchema = BlockIOSubSchemaMeta & {
  type: "array";
  items?: BlockIOSimpleTypeSubSchema;
  default?: Array<string>;
  secret?: boolean;
};

export type BlockIOStringSubSchema = BlockIOSubSchemaMeta & {
  type: "string";
  enum?: string[];
  secret?: true;
  default?: string;
  format?: string;
};

export type BlockIONumberSubSchema = BlockIOSubSchemaMeta & {
  type: "integer" | "number";
  default?: number;
  secret?: boolean;
};

export type BlockIOBooleanSubSchema = BlockIOSubSchemaMeta & {
  type: "boolean";
  default?: boolean;
  secret?: boolean;
};

export type CredentialsType = "api_key" | "oauth2" | "user_password";

export type Credentials =
  | APIKeyCredentials
  | OAuth2Credentials
  | UserPasswordCredentials;

// --8<-- [start:BlockIOCredentialsSubSchema]
export const PROVIDER_NAMES = {
  ANTHROPIC: "anthropic",
  APOLLO: "apollo",
  D_ID: "d_id",
  DISCORD: "discord",
  E2B: "e2b",
  EXA: "exa",
  FAL: "fal",
  GITHUB: "github",
  GOOGLE: "google",
  GOOGLE_MAPS: "google_maps",
  GROQ: "groq",
  HUBSPOT: "hubspot",
  IDEOGRAM: "ideogram",
  JINA: "jina",
  LINEAR: "linear",
  MEDIUM: "medium",
  MEM0: "mem0",
  NOTION: "notion",
  NVIDIA: "nvidia",
  OLLAMA: "ollama",
  OPENAI: "openai",
  OPENWEATHERMAP: "openweathermap",
  OPEN_ROUTER: "open_router",
  PINECONE: "pinecone",
  SCREENSHOTONE: "screenshotone",
  SLANT3D: "slant3d",
  SMARTLEAD: "smartlead",
  SMTP: "smtp",
  TWITTER: "twitter",
  REPLICATE: "replicate",
  REDDIT: "reddit",
  REVID: "revid",
  UNREAL_SPEECH: "unreal_speech",
  TODOIST: "todoist",
  ZEROBOUNCE: "zerobounce",
} as const;
// --8<-- [end:BlockIOCredentialsSubSchema]

export type CredentialsProviderName =
  (typeof PROVIDER_NAMES)[keyof typeof PROVIDER_NAMES];

export type BlockIOCredentialsSubSchema = BlockIOObjectSubSchema & {
  /* Mirror of backend/data/model.py:CredentialsFieldSchemaExtra */
  credentials_provider: CredentialsProviderName[];
  credentials_scopes?: string[];
  credentials_types: Array<CredentialsType>;
  discriminator?: string;
  discriminator_mapping?: { [key: string]: CredentialsProviderName };
  secret?: boolean;
};

export type BlockIONullSubSchema = BlockIOSubSchemaMeta & {
  type: "null";
  secret?: boolean;
};

// At the time of writing, combined schemas only occur on the first nested level in a
// block schema. It is typed this way to make the use of these objects less tedious.
type BlockIOCombinedTypeSubSchema = BlockIOSubSchemaMeta & { type: never } & (
    | {
        allOf: [BlockIOSimpleTypeSubSchema];
        secret?: boolean;
      }
    | {
        anyOf: BlockIOSimpleTypeSubSchema[];
        default?: string | number | boolean | null;
        secret?: boolean;
      }
    | {
        oneOf: BlockIOSimpleTypeSubSchema[];
        default?: string | number | boolean | null;
        secret?: boolean;
      }
  );

/* Mirror of backend/data/graph.py:Node */
export type Node = {
  id: string;
  block_id: string;
  input_default: { [key: string]: any };
  input_nodes: Array<{ name: string; node_id: string }>;
  output_nodes: Array<{ name: string; node_id: string }>;
  metadata: {
    position: { x: number; y: number };
    [key: string]: any;
  };
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

/* Mirror of backend/data/graph.py:GraphExecutionMeta */
export type GraphExecutionMeta = {
  execution_id: string;
  started_at: number;
  ended_at: number;
  duration: number;
  total_run_time: number;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "TERMINATED" | "FAILED";
  graph_id: GraphID;
  graph_version: number;
  preset_id?: string;
};

/* Mirror of backend/data/graph.py:GraphExecution */
export type GraphExecution = GraphExecutionMeta & {
  inputs: Record<string, any>;
  outputs: Record<string, Array<any>>;
  node_executions: NodeExecutionResult[];
};

export type GraphMeta = {
  id: GraphID;
  version: number;
  is_active: boolean;
  name: string;
  description: string;
  input_schema: GraphIOSchema;
  output_schema: GraphIOSchema;
};

export type GraphID = Brand<string, "GraphID">;

/* Derived from backend/data/graph.py:Graph._generate_schema() */
export type GraphIOSchema = {
  type: "object";
  properties: { [key: string]: GraphIOSubSchema };
  required: (keyof BlockIORootSchema["properties"])[];
};
export type GraphIOSubSchema = Omit<
  BlockIOSubSchemaMeta,
  "placeholder" | "depends_on" | "hidden"
> & {
  type: never; // bodge to avoid type checking hell; doesn't exist at runtime
  default?: string;
  secret: boolean;
};

/* Mirror of backend/data/graph.py:Graph */
export type Graph = GraphMeta & {
  nodes: Array<Node>;
  links: Array<Link>;
};

export type GraphUpdateable = Omit<
  Graph,
  "version" | "is_active" | "links" | "input_schema" | "output_schema"
> & {
  version?: number;
  is_active?: boolean;
  links: Array<LinkCreatable>;
  input_schema?: GraphIOSchema;
  output_schema?: GraphIOSchema;
};

export type GraphCreatable = Omit<GraphUpdateable, "id"> & { id?: string };

/* Mirror of backend/data/execution.py:ExecutionResult */
export type NodeExecutionResult = {
  graph_id: GraphID;
  graph_version: number;
  graph_exec_id: string;
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
  input_data: { [key: string]: any };
  output_data: { [key: string]: Array<any> };
  add_time: Date;
  queue_time?: Date;
  start_time?: Date;
  end_time?: Date;
};

/* Mirror of backend/server/v2/library/model.py:LibraryAgent */
export type LibraryAgent = {
  id: LibraryAgentID;
  agent_id: GraphID;
  agent_version: number;
  preset_id: string | null;
  updated_at: Date;
  name: string;
  description: string;
  input_schema: BlockIOObjectSubSchema;
  output_schema: BlockIOObjectSubSchema;
  is_favorite: boolean;
  is_created_by_user: boolean;
  is_latest_version: boolean;
};

export type LibraryAgentID = Brand<string, "LibraryAgentID">;

/* Mirror of backend/server/integrations/router.py:CredentialsMetaResponse */
export type CredentialsMetaResponse = {
  id: string;
  provider: CredentialsProviderName;
  type: CredentialsType;
  title?: string;
  scopes?: Array<string>;
  username?: string;
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
  title?: string;
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

// Mirror of backend/backend/data/notifications.py:NotificationType
export type NotificationType =
  | "AGENT_RUN"
  | "ZERO_BALANCE"
  | "LOW_BALANCE"
  | "BLOCK_EXECUTION_FAILED"
  | "CONTINUOUS_AGENT_ERROR"
  | "DAILY_SUMMARY"
  | "WEEKLY_SUMMARY"
  | "MONTHLY_SUMMARY";

// Mirror of backend/backend/data/notifications.py:NotificationPreference
export type NotificationPreferenceDTO = {
  email: string;
  preferences: { [key in NotificationType]: boolean };
  daily_limit: number;
};

export type NotificationPreference = NotificationPreferenceDTO & {
  user_id: string;
  emails_sent_today: number;
  last_reset_date: Date;
};

/* Mirror of backend/data/integrations.py:Webhook */
export type Webhook = {
  id: string;
  url: string;
  provider: CredentialsProviderName;
  credentials_id: string;
  webhook_type: string;
  resource?: string;
  events: string[];
  secret: string;
  config: Record<string, any>;
  provider_webhook_id?: string;
};

export type User = {
  id: string;
  email: string;
};

export enum BlockUIType {
  STANDARD = "Standard",
  INPUT = "Input",
  OUTPUT = "Output",
  NOTE = "Note",
  WEBHOOK = "Webhook",
  WEBHOOK_MANUAL = "Webhook (manual)",
  AGENT = "Agent",
}

export enum SpecialBlockID {
  AGENT = "e189baac-8c20-45a1-94a7-55177ea42565",
  INPUT = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
  OUTPUT = "363ae599-353e-4804-937e-b2ee3cef3da4",
}

export type AnalyticsMetrics = {
  metric_name: string;
  metric_value: number;
  data_string: string;
};

export type AnalyticsDetails = {
  type: string;
  data: { [key: string]: any };
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
  image_urls: string[];
  date_submitted: string;
  status: string;
  runs: number;
  rating: number;
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
  categories: string[];
};

export type ProfileDetails = {
  name: string;
  username: string;
  description: string;
  links: string[];
  avatar_url: string;
};

export type Schedule = {
  id: string;
  name: string;
  cron: string;
  user_id: string;
  graph_id: GraphID;
  graph_version: number;
  input_data: { [key: string]: any };
  next_run_time: Date;
};

export type ScheduleCreatable = {
  cron: string;
  graph_id: GraphID;
  graph_version: number;
  input_data: { [key: string]: any };
};

export type MyAgent = {
  agent_id: string;
  agent_version: number;
  agent_name: string;
  last_edited: string;
  description: string;
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
  transaction_type: string;
  amount: number;
  balance: number;
  description: string;
  usage_graph_id: GraphID;
  usage_execution_id: string;
  usage_node_count: number;
  usage_starting_time: Date;
}

export interface TransactionHistory {
  transactions: CreditTransaction[];
  next_transaction_time: Date | null;
}

export interface RefundRequest {
  id: string;
  user_id: string;
  transaction_key: string;
  amount: number;
  reason: string;
  result: string | null;
  status: string;
  created_at: Date;
  updated_at: Date;
}

/* *** UTILITIES *** */

/** Use branded types for IDs -> deny mixing IDs between different object classes */
export type Brand<T, Brand extends string> = T & {
  readonly [B in Brand as `__${B}_brand`]: never;
};
