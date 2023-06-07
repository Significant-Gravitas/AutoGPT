---
description: 'Require both operands of addition to be the same type and be `bigint`, `number`, or `string`.'
---

> 🛑 This file is source code, not the primary documentation location! 🛑
>
> See **https://typescript-eslint.io/rules/restrict-plus-operands** for documentation.

TypeScript allows `+` adding together two values of any type(s).
However, adding values that are not the same type and/or are not the same primitive type is often a sign of programmer error.

This rule reports when a `+` operation combines two values of different types, or a type that is not `bigint`, `number`, or `string`.

## Examples

<!--tabs-->

### ❌ Incorrect

```ts
var foo = '5.5' + 5;
var foo = 1n + 1;
```

### ✅ Correct

```ts
var foo = parseInt('5.5', 10) + 10;
var foo = 1n + 1n;
```

## Options

### `checkCompoundAssignments`

Examples of code for this rule with `{ checkCompoundAssignments: true }`:

<!--tabs-->

#### ❌ Incorrect

```ts
/*eslint @typescript-eslint/restrict-plus-operands: ["error", { "checkCompoundAssignments": true }]*/

let foo: string | undefined;
foo += 'some data';

let bar: string = '';
bar += 0;
```

#### ✅ Correct

```ts
/*eslint @typescript-eslint/restrict-plus-operands: ["error", { "checkCompoundAssignments": true }]*/

let foo: number = 0;
foo += 1;

let bar = '';
bar += 'test';
```

### `allowAny`

Examples of code for this rule with `{ allowAny: true }`:

<!--tabs-->

#### ❌ Incorrect

```ts
var fn = (a: any, b: boolean) => a + b;
var fn = (a: any, b: []) => a + b;
var fn = (a: any, b: {}) => a + b;
```

#### ✅ Correct

```ts
var fn = (a: any, b: any) => a + b;
var fn = (a: any, b: string) => a + b;
var fn = (a: any, b: bigint) => a + b;
var fn = (a: any, b: number) => a + b;
```

## When Not To Use It

If you don't mind `"[object Object]"` in your strings, then you will not need this rule.

## Related To

- [`no-base-to-string`](./no-base-to-string.md)
- [`restrict-template-expressions`](./restrict-template-expressions.md)

## Further Reading

- [`Object.prototype.toString()` MDN documentation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/toString)
