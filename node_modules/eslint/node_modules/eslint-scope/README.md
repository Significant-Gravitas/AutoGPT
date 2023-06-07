[![npm version](https://img.shields.io/npm/v/eslint-scope.svg)](https://www.npmjs.com/package/eslint-scope)
[![Downloads](https://img.shields.io/npm/dm/eslint-scope.svg)](https://www.npmjs.com/package/eslint-scope)
[![Build Status](https://github.com/eslint/eslint-scope/workflows/CI/badge.svg)](https://github.com/eslint/eslint-scope/actions)

# ESLint Scope

ESLint Scope is the [ECMAScript](http://www.ecma-international.org/publications/standards/Ecma-262.htm) scope analyzer used in ESLint. It is a fork of [escope](http://github.com/estools/escope).

## Install

```
npm i eslint-scope --save
```

## 📖 Usage

To use in an ESM file:

```js
import * as eslintScope from 'eslint-scope';
```

To use in a CommonJS file:

```js
const eslintScope = require('eslint-scope');
```

Example:

```js
import * as eslintScope from 'eslint-scope';
import * as espree from 'espree';
import estraverse from 'estraverse';

const ast = espree.parse(code, { range: true });
const scopeManager = eslintScope.analyze(ast);

const currentScope = scopeManager.acquire(ast);   // global scope

estraverse.traverse(ast, {
    enter (node, parent) {
        // do stuff

        if (/Function/.test(node.type)) {
            currentScope = scopeManager.acquire(node);  // get current function scope
        }
    },
    leave(node, parent) {
        if (/Function/.test(node.type)) {
            currentScope = currentScope.upper;  // set to parent scope
        }

        // do stuff
    }
});
```

## Contributing

Issues and pull requests will be triaged and responded to as quickly as possible. We operate under the [ESLint Contributor Guidelines](http://eslint.org/docs/developer-guide/contributing), so please be sure to read them before contributing. If you're not sure where to dig in, check out the [issues](https://github.com/eslint/eslint-scope/issues).

## Build Commands

* `npm test` - run all linting and tests
* `npm run lint` - run all linting

## License

ESLint Scope is licensed under a permissive BSD 2-clause license.
