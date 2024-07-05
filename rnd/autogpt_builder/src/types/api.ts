import {XYPosition} from "reactflow";
import {STATUS_COMPLETED, STATUS_FAILED, STATUS_INCOMPLETE, STATUS_QUEUED, STATUS_RUNNING} from "@/constants/constants";

export type ObjectSchema = {
    type: string;
    properties: { [key: string]: any };
    additionalProperties?: { type: string };
    required?: string[];
  };


/* Mirror of autogpt_server/data/block.py:Block */
export type Block = {
  id: string;
  name: string;
  description: string;
  inputSchema: ObjectSchema;
  outputSchema: ObjectSchema;
};

/* Mirror of autogpt_server/data/graph.py:Node */
export type Node = {
  id: string;
  block_id: string;
  input_default: Map<string, any>;
  input_nodes: Array<{ name: string, node_id: string }>;
  output_nodes: Array<{ name: string, node_id: string }>;
  metadata: {
    position: XYPosition;
    [key: string]: any;
  };
};

/* Mirror of autogpt_server/data/graph.py:Link */
export type Link = {
  source_id: string;
  sink_id: string;
  source_name: string;
  sink_name: string;
}

/* Mirror of autogpt_server/data/graph.py:Graph */
export type Flow = {
  id: string;
  name: string;
  description: string;
  nodes: Array<Node>;
  links: Array<Link>;
};

export type FlowCreateBody = Flow | {
  id?: string;
}

/* Derived from autogpt_server/executor/manager.py:ExecutionManager.add_execution */
export type FlowExecuteResponse = {
  /* ID of the initiated run */
  id: string;
  /* List of node executions */
  executions: Array<{ id: string, node_id: string }>;
};

/* Mirror of autogpt_server/data/execution.py:ExecutionResult */
export type NodeExecutionResult = {
  graph_exec_id: string;
  node_exec_id: string;
  node_id: string;
  status: typeof STATUS_INCOMPLETE | typeof STATUS_QUEUED | typeof STATUS_RUNNING | typeof STATUS_COMPLETED | typeof STATUS_FAILED;
  input_data: Map<string, any>;
  output_data: Map<string, any[]>;
  add_time: Date;
  queue_time?: Date;
  start_time?: Date;
  end_time?: Date;
};