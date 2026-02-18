import { RJSFSchema } from "@rjsf/utils";

type OutputSchemaProperties = Record<string, RJSFSchema>;

type OutputTreeNode = {
  fullKey: string;
  title: string;
  schema: RJSFSchema;
  isTopLevel: boolean;
  isObject: boolean;
  children: OutputTreeNode[];
};

export type VisibleOutputTreeNode = {
  fullKey: string;
  title: string;
  schema: RJSFSchema;
  isTopLevel: boolean;
  isObject: boolean;
  isCollapsed: boolean;
  isConnected: boolean;
  hasConnectedDescendant: boolean;
  isVisible: boolean;
  totalChildrenCount: number;
  hiddenChildrenCount: number;
  children: VisibleOutputTreeNode[];
};

function getSchemaProperties(schema: RJSFSchema): OutputSchemaProperties {
  if (!schema || !schema.properties) {
    return {};
  }

  return schema.properties as OutputSchemaProperties;
}

function isObjectOutput(schema: RJSFSchema): boolean {
  return schema.type === "object" || Boolean(schema.properties);
}

export function buildOutputTree(
  properties: OutputSchemaProperties,
  keyPrefix = "",
  titlePrefix = "",
  isTopLevel = true,
): OutputTreeNode[] {
  return Object.entries(properties).map(([key, fieldSchema]) => {
    const fullKey = keyPrefix ? `${keyPrefix}_#_${key}` : key;
    const fieldTitle = `${titlePrefix}${fieldSchema.title || key}`;
    const childProperties = getSchemaProperties(fieldSchema);

    return {
      fullKey,
      title: fieldTitle,
      schema: fieldSchema,
      isTopLevel,
      isObject: isObjectOutput(fieldSchema),
      children: buildOutputTree(
        childProperties,
        fullKey,
        `${fieldTitle}.`,
        false,
      ),
    };
  });
}

function buildVisibleNode(
  node: OutputTreeNode,
  isHandleConnected: (handleId: string) => boolean,
  isCollapsed: (handleId: string) => boolean,
  isHiddenByCollapsedAncestor: boolean,
): VisibleOutputTreeNode {
  const nodeIsConnected = isHandleConnected(node.fullKey);
  const nodeIsCollapsed = node.isObject ? isCollapsed(node.fullKey) : false;

  const childHiddenByCollapsedAncestor =
    isHiddenByCollapsedAncestor || (node.isObject && nodeIsCollapsed);

  const children = node.children.map((child) =>
    buildVisibleNode(
      child,
      isHandleConnected,
      isCollapsed,
      childHiddenByCollapsedAncestor,
    ),
  );

  const hasConnectedDescendant = children.some(
    (child) => child.isConnected || child.hasConnectedDescendant,
  );

  const isVisible =
    node.isTopLevel ||
    !isHiddenByCollapsedAncestor ||
    nodeIsConnected ||
    hasConnectedDescendant;

  const visibleChildren = children.filter((child) => child.isVisible);

  return {
    fullKey: node.fullKey,
    title: node.title,
    schema: node.schema,
    isTopLevel: node.isTopLevel,
    isObject: node.isObject,
    isCollapsed: nodeIsCollapsed,
    isConnected: nodeIsConnected,
    hasConnectedDescendant,
    isVisible,
    totalChildrenCount: children.length,
    hiddenChildrenCount: children.length - visibleChildren.length,
    children: visibleChildren,
  };
}

export function buildVisibleOutputTree({
  properties,
  isHandleConnected,
  isCollapsed,
}: {
  properties: OutputSchemaProperties;
  isHandleConnected: (handleId: string) => boolean;
  isCollapsed: (handleId: string) => boolean;
}): VisibleOutputTreeNode[] {
  const outputTree = buildOutputTree(properties);
  return outputTree
    .map((node) =>
      buildVisibleNode(node, isHandleConnected, isCollapsed, false),
    )
    .filter((node) => node.isVisible);
}
