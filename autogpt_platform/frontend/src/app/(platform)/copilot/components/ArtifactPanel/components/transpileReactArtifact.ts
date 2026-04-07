function ensureJsxExtension(filename: string): string {
  // TypeScript infers JSX parsing from the file extension; if the artifact
  // title is "component" or "foo.ts", TSX syntax in the source will be
  // treated as a syntax error. Force a .tsx extension for transpilation.
  const lower = filename.toLowerCase();
  if (lower.endsWith(".tsx") || lower.endsWith(".jsx")) return filename;
  return `${filename || "artifact"}.tsx`;
}

export async function transpileReactArtifactSource(
  source: string,
  filename: string,
) {
  const ts = await import("typescript");
  const result = ts.transpileModule(source, {
    compilerOptions: {
      allowJs: true,
      esModuleInterop: true,
      jsx: ts.JsxEmit.React,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
    },
    fileName: ensureJsxExtension(filename),
    reportDiagnostics: true,
  });

  const diagnostics =
    result.diagnostics?.filter(
      (diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error,
    ) ?? [];

  if (diagnostics.length > 0) {
    const message = diagnostics
      .slice(0, 3)
      .map((diagnostic) =>
        ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
      )
      .join("\n\n");
    throw new Error(message);
  }

  return result.outputText;
}
