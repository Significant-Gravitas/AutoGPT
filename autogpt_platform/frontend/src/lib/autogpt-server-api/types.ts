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
  costs: BlockCost[];
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

type BlockIOSimpleTypeSubSchema =
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
};

export type BlockIOObjectSubSchema = BlockIOSubSchemaMeta & {
  type: "object";
  properties: { [key: string]: BlockIOSubSchema };
  default?: { [key: keyof BlockIOObjectSubSchema["properties"]]: any };
  required?: (keyof BlockIOObjectSubSchema["properties"])[];
};

export type BlockIOKVSubSchema = BlockIOSubSchemaMeta & {
  type: "object";
  additionalProperties: { type: "string" | "number" | "integer" };
  default?: { [key: string]: string | number };
};

export type BlockIOArraySubSchema = BlockIOSubSchemaMeta & {
  type: "array";
  items?: BlockIOSimpleTypeSubSchema;
  default?: Array<string>;
};

export type BlockIOStringSubSchema = BlockIOSubSchemaMeta & {
  type: "string";
  enum?: string[];
  secret?: true;
  default?: string;
};

export type BlockIONumberSubSchema = BlockIOSubSchemaMeta & {
  type: "integer" | "number";
  default?: number;
};

export type BlockIOBooleanSubSchema = BlockIOSubSchemaMeta & {
  type: "boolean";
  default?: boolean;
};

export type CredentialsType = "api_key" | "oauth2";

// --8<-- [start:BlockIOCredentialsSubSchema]
export type BlockIOCredentialsSubSchema = BlockIOSubSchemaMeta & {
  credentials_provider: "github" | "google" | "notion";
  credentials_scopes?: string[];
  credentials_types: Array<CredentialsType>;
};
// --8<-- [end:BlockIOCredentialsSubSchema]

export type BlockIONullSubSchema = BlockIOSubSchemaMeta & {
  type: "null";
};

// At the time of writing, combined schemas only occur on the first nested level in a
// block schema. It is typed this way to make the use of these objects less tedious.
type BlockIOCombinedTypeSubSchema = BlockIOSubSchemaMeta &
  (
    | {
        allOf: [BlockIOSimpleTypeSubSchema];
      }
    | {
        anyOf: BlockIOSimpleTypeSubSchema[];
        default?: string | number | boolean | null;
      }
    | {
        oneOf: BlockIOSimpleTypeSubSchema[];
        default?: string | number | boolean | null;
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

/* Mirror of autogpt_server/data/graph.py:ExecutionMeta */
export type ExecutionMeta = {
  execution_id: string;
  started_at: number;
  ended_at: number;
  duration: number;
  total_run_time: number;
  status: "running" | "waiting" | "success" | "failed";
};

/* Mirror of backend/data/graph.py:GraphMeta */
export type GraphMeta = {
  id: string;
  version: number;
  is_active: boolean;
  is_template: boolean;
  name: string;
  description: string;
};

export type GraphMetaWithRuns = GraphMeta & {
  executions: ExecutionMeta[];
};

/* Mirror of backend/data/graph.py:Graph */
export type Graph = GraphMeta & {
  nodes: Array<Node>;
  links: Array<Link>;
};

export type GraphUpdateable = Omit<
  Graph,
  "version" | "is_active" | "is_template" | "links"
> & {
  version?: number;
  is_active?: boolean;
  is_template?: boolean;
  links: Array<LinkCreatable>;
};

export type GraphCreatable = Omit<GraphUpdateable, "id"> & { id?: string };

/* Derived from backend/executor/manager.py:ExecutionManager.add_execution */
export type GraphExecuteResponse = {
  /** ID of the initiated run */
  id: string;
  /** List of node executions */
  executions: Array<{ id: string; node_id: string }>;
};

/* Mirror of backend/data/execution.py:ExecutionResult */
export type NodeExecutionResult = {
  graph_exec_id: string;
  node_exec_id: string;
  graph_id: string;
  graph_version: number;
  node_id: string;
  status: "INCOMPLETE" | "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
  input_data: { [key: string]: any };
  output_data: { [key: string]: Array<any> };
  add_time: Date;
  queue_time?: Date;
  start_time?: Date;
  end_time?: Date;
};

/* Mirror of backend/server/integrations.py:CredentialsMetaResponse */
export type CredentialsMetaResponse = {
  id: string;
  type: CredentialsType;
  title?: string;
  scopes?: Array<string>;
  username?: string;
};

/* Mirror of backend/data/model.py:CredentialsMetaInput */
export type CredentialsMetaInput = {
  id: string;
  type: CredentialsType;
  title?: string;
  provider: string;
};

/* Mirror of autogpt_libs/supabase_integration_credentials_store/types.py:_BaseCredentials */
type BaseCredentials = {
  id: string;
  type: CredentialsType;
  title?: string;
  provider: string;
};

/* Mirror of autogpt_libs/supabase_integration_credentials_store/types.py:OAuth2Credentials */
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

/* Mirror of autogpt_libs/supabase_integration_credentials_store/types.py:APIKeyCredentials */
export type APIKeyCredentials = BaseCredentials & {
  type: "api_key";
  title: string;
  api_key: string;
  expires_at?: number;
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
