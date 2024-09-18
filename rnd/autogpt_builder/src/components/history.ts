// history.ts
import { CustomNodeData } from "./CustomNode";
import { CustomEdgeData } from "./CustomEdge";
import { Edge } from "@xyflow/react";

type ActionType =
  | "ADD_NODE"
  | "DELETE_NODE"
  | "ADD_EDGE"
  | "DELETE_EDGE"
  | "UPDATE_NODE"
  | "MOVE_NODE"
  | "UPDATE_INPUT"
  | "UPDATE_NODE_POSITION";

type AddNodePayload = { node: CustomNodeData };
type DeleteNodePayload = { nodeId: string };
type AddEdgePayload = { edge: Edge<CustomEdgeData> };
type DeleteEdgePayload = { edgeId: string };
type UpdateNodePayload = { nodeId: string; newData: Partial<CustomNodeData> };
type MoveNodePayload = { nodeId: string; position: { x: number; y: number } };
type UpdateInputPayload = {
  nodeId: string;
  oldValues: { [key: string]: any };
  newValues: { [key: string]: any };
};
type UpdateNodePositionPayload = {
  nodeId: string;
  oldPosition: { x: number; y: number };
  newPosition: { x: number; y: number };
};

type ActionPayload =
  | AddNodePayload
  | DeleteNodePayload
  | AddEdgePayload
  | DeleteEdgePayload
  | UpdateNodePayload
  | MoveNodePayload
  | UpdateInputPayload
  | UpdateNodePositionPayload;

type Action = {
  type: ActionType;
  payload: ActionPayload;
  undo: () => void;
  redo: () => void;
};

class History {
  private past: Action[] = [];
  private future: Action[] = [];

  push(action: Action) {
    this.past.push(action);
    this.future = [];
  }

  undo() {
    const action = this.past.pop();
    if (action) {
      action.undo();
      this.future.push(action);
    }
  }

  redo() {
    const action = this.future.pop();
    if (action) {
      action.redo();
      this.past.push(action);
    }
  }

  canUndo(): boolean {
    return this.past.length > 0;
  }

  canRedo(): boolean {
    return this.future.length > 0;
  }

  clear() {
    this.past = [];
    this.future = [];
  }

  getHistoryState() {
    return {
      past: [...this.past],
      future: [...this.future],
    };
  }
}

export const history = new History();
