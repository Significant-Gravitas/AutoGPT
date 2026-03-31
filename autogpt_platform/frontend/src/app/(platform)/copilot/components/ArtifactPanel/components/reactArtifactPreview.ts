function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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
    fileName: filename,
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

export function collectPreviewStyles() {
  if (typeof document === "undefined") {
    return "";
  }

  return Array.from(document.querySelectorAll('link[rel="stylesheet"], style'))
    .map((node) => node.outerHTML)
    .join("\n");
}

export function buildReactArtifactSrcDoc(
  compiledCode: string,
  title: string,
  stylesMarkup: string,
) {
  const safeTitle = escapeHtml(title);
  const runtime = JSON.stringify(compiledCode).replace(/</g, "\\u003c");

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${safeTitle}</title>
    ${stylesMarkup}
    <style>
      html, body, #root {
        height: 100%;
        margin: 0;
      }

      body {
        background:
          radial-gradient(circle at top, rgba(148, 163, 184, 0.18), transparent 35%),
          #f8fafc;
        color: #18181b;
        font-family: ui-sans-serif, system-ui, sans-serif;
      }

      #root {
        box-sizing: border-box;
        min-height: 100%;
        isolation: isolate;
      }

      #error {
        display: none;
        box-sizing: border-box;
        margin: 24px;
        padding: 16px;
        border: 1px solid #fecaca;
        border-radius: 16px;
        background: #fff1f2;
        color: #991b1b;
        font-family: ui-monospace, SFMono-Regular, monospace;
        white-space: pre-wrap;
      }
    </style>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  </head>
  <body>
    <div id="root"></div>
    <div id="error"></div>
    <script>
      (function () {
        const compiledCode = ${runtime};
        const rootElement = document.getElementById("root");
        const errorElement = document.getElementById("error");

        function showError(error) {
          rootElement.style.display = "none";
          errorElement.style.display = "block";
          errorElement.textContent =
            error instanceof Error && error.stack
              ? error.stack
              : error instanceof Error
                ? error.message
                : String(error);
        }

        function getModuleExports(module, exports) {
          return {
            ...exports,
            ...(typeof module.exports === "object" ? module.exports : {}),
          };
        }

        function getRenderableCandidate(moduleExports) {
          if (typeof moduleExports.default === "function") {
            return moduleExports.default;
          }

          if (typeof moduleExports.App === "function") {
            return moduleExports.App;
          }

          const namedCandidate = Object.entries(moduleExports).find(
            ([name, value]) =>
              name !== "default" &&
              !name.endsWith("Provider") &&
              /^[A-Z]/.test(name) &&
              typeof value === "function",
          );

          if (namedCandidate) {
            return namedCandidate[1];
          }

          if (typeof App !== "undefined" && typeof App === "function") {
            return App;
          }

          throw new Error(
            "No renderable component found. Export a default component, export App, or export a named component.",
          );
        }

        function wrapWithProviders(Component, moduleExports) {
          const providers = Object.entries(moduleExports)
            .filter(
              ([name, value]) =>
                name !== "default" &&
                name.endsWith("Provider") &&
                typeof value === "function",
            )
            .map(([, value]) => value);

          if (providers.length === 0) {
            return Component;
          }

          return function WrappedArtifactPreview() {
            let tree = React.createElement(Component);

            for (let i = providers.length - 1; i >= 0; i -= 1) {
              tree = React.createElement(providers[i], null, tree);
            }

            return tree;
          };
        }

        function require(name) {
          if (name === "react") {
            return React;
          }

          if (name === "react-dom") {
            return ReactDOM;
          }

          if (name === "react-dom/client") {
            return { createRoot: ReactDOM.createRoot };
          }

          if (name === "react/jsx-runtime") {
            return {
              Fragment: React.Fragment,
              jsx: React.createElement,
              jsxs: React.createElement,
            };
          }

          throw new Error("Unsupported import in artifact preview: " + name);
        }

        class PreviewErrorBoundary extends React.Component {
          constructor(props) {
            super(props);
            this.state = { error: null };
          }

          static getDerivedStateFromError(error) {
            return { error };
          }

          render() {
            if (this.state.error) {
              return React.createElement(
                "div",
                {
                  style: {
                    margin: "24px",
                    padding: "16px",
                    border: "1px solid #fecaca",
                    borderRadius: "16px",
                    background: "#fff1f2",
                    color: "#991b1b",
                    fontFamily: "ui-monospace, SFMono-Regular, monospace",
                    whiteSpace: "pre-wrap",
                  },
                },
                this.state.error.stack || this.state.error.message || String(this.state.error),
              );
            }

            return this.props.children;
          }
        }

        try {
          const exports = {};
          const module = { exports };
          const factory = new Function(
            "React",
            "ReactDOM",
            "module",
            "exports",
            "require",
            \`
              "use strict";
              \${compiledCode}
              return {
                module,
                exports,
                app: typeof App !== "undefined" ? App : undefined,
              };
            \`,
          );

          const executionResult = factory(
            React,
            ReactDOM,
            module,
            exports,
            require,
          );
          const moduleExports = getModuleExports(
            executionResult.module,
            executionResult.exports,
          );

          if (
            executionResult.app &&
            typeof moduleExports.App !== "function"
          ) {
            moduleExports.App = executionResult.app;
          }

          const Component = wrapWithProviders(
            getRenderableCandidate(moduleExports),
            moduleExports,
          );

          ReactDOM.createRoot(rootElement).render(
            React.createElement(
              PreviewErrorBoundary,
              null,
              React.createElement(Component),
            ),
          );
        } catch (error) {
          showError(error);
        }
      })();
    </script>
  </body>
</html>`;
}
