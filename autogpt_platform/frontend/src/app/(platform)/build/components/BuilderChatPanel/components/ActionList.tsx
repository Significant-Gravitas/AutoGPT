import type { CustomNode } from "../../FlowEditor/nodes/CustomNode/CustomNode";
import { GraphAction, getActionKey, getNodeDisplayName } from "../helpers";

interface ActionListProps {
  parsedActions: GraphAction[];
  nodes: CustomNode[];
  appliedActionKeys: Set<string>;
  onApplyAction: (action: GraphAction) => void;
}

export function ActionList({
  parsedActions,
  nodes,
  appliedActionKeys,
  onApplyAction,
}: ActionListProps) {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  return (
    <div className="space-y-2 rounded-lg border border-violet-100 bg-violet-50 p-3">
      <p className="text-xs font-medium text-violet-700">Suggested changes</p>
      {parsedActions.map((action) => {
        const key = getActionKey(action);
        return (
          <ActionItem
            key={key}
            action={action}
            nodeMap={nodeMap}
            isApplied={appliedActionKeys.has(key)}
            onApply={onApplyAction}
          />
        );
      })}
    </div>
  );
}

interface ActionItemProps {
  action: GraphAction;
  nodeMap: Map<string, CustomNode>;
  isApplied: boolean;
  onApply: (action: GraphAction) => void;
}

function ActionItem({ action, nodeMap, isApplied, onApply }: ActionItemProps) {
  const label =
    action.type === "update_node_input"
      ? `Set "${getNodeDisplayName(nodeMap.get(action.nodeId), action.nodeId)}" "${action.key}" = ${JSON.stringify(action.value)}`
      : `Connect "${getNodeDisplayName(nodeMap.get(action.source), action.source)}" → "${getNodeDisplayName(nodeMap.get(action.target), action.target)}"`;

  return (
    <div className="flex items-start justify-between gap-2 rounded bg-white p-2 text-xs shadow-sm">
      <span className="leading-tight text-slate-700">{label}</span>
      {isApplied ? (
        <span
          role="status"
          aria-live="polite"
          className="shrink-0 rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700"
        >
          Applied
        </span>
      ) : (
        <button
          type="button"
          onClick={() => onApply(action)}
          aria-label={`Apply: ${label}`}
          className="shrink-0 rounded bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-700 hover:bg-violet-200"
        >
          Apply
        </button>
      )}
    </div>
  );
}
