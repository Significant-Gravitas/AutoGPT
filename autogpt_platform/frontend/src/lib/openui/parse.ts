import { createParser } from "@openuidev/lang-core";
import { catalog, MAX_SOURCE_LENGTH } from "./catalog";
import { validateWorkspace } from "./validate";
import { validateSource } from "./source-validation";
import { validateConnections } from "./validate-connections";
import { workspaceSummary } from "./workspace-summary";

export function parseWorkspace(source: string) {
  if (source.length > MAX_SOURCE_LENGTH)
    throw new Error("This workspace is too large. Try a more focused request.");
  validateSource(source);
  const result = createParser(catalog.toJSONSchema()).parse(source);
  if (result.meta.incomplete)
    throw new Error("Incomplete source. Return a complete Workspace program.");
  if (result.meta.unresolved.length)
    throw new Error(
      `Unresolved references: ${result.meta.unresolved.slice(0, 5).join(", ")}. Define each referenced section.`,
    );
  if (result.meta.errors.length)
    throw new Error(
      result.meta.errors
        .slice(0, 3)
        .map((issue) => `${issue.component}${issue.path}: ${issue.message}`)
        .join("; "),
    );
  if (result.queryStatements.length || result.mutationStatements.length)
    throw new Error(
      "Query and Mutation are not supported. Retrieve data through Copilot tools before rendering.",
    );
  if (!result.root || result.root.typeName !== "Workspace")
    throw new Error(
      "Start with root = Workspace(title, description, sections).",
    );
  validateWorkspace(result.root);
  validateConnections(result.root);
  return result.root;
}

export function workspaceText(source: string): string {
  try {
    const root = parseWorkspace(source);
    return workspaceSummary(root);
  } catch {
    return "A text summary will be available when this workspace finishes rendering.";
  }
}
