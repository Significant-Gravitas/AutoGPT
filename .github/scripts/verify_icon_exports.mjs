import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";

const require = createRequire(path.resolve("package.json"));
const ts = require("typescript");
const packageName = "@hugeicons/core-free-icons";
const packagePath = path.dirname(require.resolve(`${packageName}/package.json`));
const root = await import(pathToFileURL(path.join(packagePath, "dist/esm/index.js")));
const rootSource = ts.createSourceFile("index.js", readFileSync(path.join(packagePath, "dist/esm/index.js"), "utf8"), ts.ScriptTarget.Latest, true);
const canonicalNames = new Map();
for (const statement of rootSource.statements) {
  if (ts.isExportDeclaration(statement) && statement.exportClause && ts.isNamedExports(statement.exportClause)) {
    for (const element of statement.exportClause.elements) {
      const canonical = statement.moduleSpecifier && ts.isStringLiteral(statement.moduleSpecifier)
        ? path.basename(statement.moduleSpecifier.text, ".js")
        : (element.propertyName ?? element.name).text;
      canonicalNames.set(element.name.text, canonical);
    }
  }
}
const files = execFileSync("git", ["ls-files", "-z", "--", "*.ts", "*.tsx", "*.mts"], { encoding: "utf8" }).split("\0").filter(Boolean);
const names = new Set();
for (const file of files) {
  const source = ts.createSourceFile(file, readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true);
  function visit(node) {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier) && node.moduleSpecifier.text === packageName) {
      const clause = node.importClause;
      if (clause?.isTypeOnly) return;
      assert(clause && !clause.name && clause.namedBindings && ts.isNamedImports(clause.namedBindings), `Unsupported value import in ${file}`);
      for (const element of clause.namedBindings.elements) {
        if (!element.isTypeOnly) names.add((element.propertyName ?? element.name).text);
      }
    }
    if (ts.isExportDeclaration(node) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier) && node.moduleSpecifier.text === packageName && !node.isTypeOnly) {
      assert(node.exportClause && ts.isNamedExports(node.exportClause), `Unsupported value re-export in ${file}`);
      for (const element of node.exportClause.elements) {
        if (!element.isTypeOnly) names.add((element.propertyName ?? element.name).text);
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}
assert(names.size > 0, "No icon imports found");
const transform = {};
for (const name of names) {
  assert.match(name, /^[A-Za-z_$][A-Za-z0-9_$]*$/);
  const canonical = canonicalNames.get(name) ?? name;
  assert(root[name], `No root export for ${name}`);
  assert.match(canonical, /^[A-Za-z_$][A-Za-z0-9_$]*$/);
  const direct = await import(pathToFileURL(path.join(packagePath, `dist/esm/${canonical}.js`)));
  assert.deepStrictEqual(direct.default, root[name], `Icon differs: ${name}`);
  transform[`^${name}$`] = `${packageName}/${canonical}`;
}
assert(process.argv[2], "An output path is required");
writeFileSync(process.argv[2], JSON.stringify(transform, null, 2) + "\n");
console.log(JSON.stringify({ version: require(`${packageName}/package.json`).version, matchedIcons: names.size, names: [...names].sort() }, null, 2));
