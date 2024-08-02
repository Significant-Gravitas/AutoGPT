/* Mirror of autogpt_server/data/block.py:Block */
export type Block = {
  id: string;
  name: string;
  description: string;
  inputSchema: BlockIORootSchema;
  outputSchema: BlockIORootSchema;
};

export type BlockIORootSchema = {
  type: "object";
  properties: { [key: string]: BlockIOSchema };
  required?: string[];
  additionalProperties?: { type: string };
}

export type BlockIOSchema = {
  title?: string;
  description?: string;
  placeholder?: string;
} & (BlockIOSimpleTypeSchema | BlockIOCombinedTypeSchema);

type BlockIOSimpleTypeSchema = {
  type: "object";
  properties: { [key: string]: BlockIOSchema };
  required?: string[];
  additionalProperties?: { type: string };
} | {
  type: "array";
  items?: BlockIOSimpleTypeSchema;
} | {
  type: "string";
  enum?: string[];
  secret?: true;
  default?: string;
} | {
  type: "integer" | "number";
  default?: number;
} | {
  type: "boolean";
  default?: boolean;
} | {
  type: "null";
};

// At the time of writing, combined schemas only occur on the first nested level in a
// block schema. It is typed this way to make the use of these objects less tedious.
type BlockIOCombinedTypeSchema = {
  allOf: [BlockIOSimpleTypeSchema];
} | {
  anyOf: BlockIOSimpleTypeSchema[];
  default?: string | number | boolean | null;
} | {
  oneOf: BlockIOSimpleTypeSchema[];
  default?: string | number | boolean | null;
};

/* Mirror of autogpt_server/data/graph.py:Node */
export type Node = {
  id: string;
  block_id: string;
  input_default: { [key: string]: any };
  input_nodes: Array<{ name: string, node_id: string }>;
  output_nodes: Array<{ name: string, node_id: string }>;
  metadata: {
    position: { x: number; y: number; };
    [key: string]: any;
  };
};

/* Mirror of autogpt_server/data/graph.py:Link */
export type Link = {
  id: string;
  source_id: string;
  sink_id: string;
  source_name: string;
  sink_name: string;
}

export type LinkCreatable = Omit<Link, "id"> & {
  id?: string;
}

/* Mirror of autogpt_server/data/graph.py:GraphMeta */
export type GraphMeta = {
  id: string;
  version: number;
  is_active: boolean;
  is_template: boolean;
  name: string;
  description: string;
}

/* Mirror of autogpt_server/data/graph.py:Graph */
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
}

export type GraphCreatable = Omit<GraphUpdateable, "id"> & { id?: string }

/* Derived from autogpt_server/executor/manager.py:ExecutionManager.add_execution */
export type GraphExecuteResponse = {
  /** ID of the initiated run */
  id: string;
  /** List of node executions */
  executions: Array<{ id: string, node_id: string }>;
};

/* Mirror of autogpt_server/data/execution.py:ExecutionResult */
export type NodeExecutionResult = {
  graph_exec_id: string;
  node_exec_id: string;
  graph_id: string;
  graph_version: number;
  node_id: string;
  status: 'INCOMPLETE' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  input_data: { [key: string]: any };
  output_data: { [key: string]: Array<any> };
  add_time: Date;
  queue_time?: Date;
  start_time?: Date;
  end_time?: Date;
};
