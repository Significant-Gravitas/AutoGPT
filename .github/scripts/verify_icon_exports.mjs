import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";

const require = createRequire(path.resolve("package.json"));
const ts = require("typescript");
const packageName = "@hugeicons/core-free-icons";
const packagePath = path.dirname(require.resolve(`${packageName}/package.json`));
const root = await import(pathToFileURL(path.join(packagePath, "dist/esm/index.js")));
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
for (const name of names) {
  assert.match(name, /^[A-Za-z_$][A-Za-z0-9_$]*$/);
  const direct = await import(pathToFileURL(path.join(packagePath, `dist/esm/${name}.js`)));
  assert.deepStrictEqual(direct.default, root[name], `Icon differs: ${name}`);
}
console.log(JSON.stringify({ version: require(`${packageName}/package.json`).version, matchedIcons: names.size, names: [...names].sort() }, null, 2));
