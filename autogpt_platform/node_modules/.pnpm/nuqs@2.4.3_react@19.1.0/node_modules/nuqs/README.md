# nuqs

[![NPM](https://img.shields.io/npm/v/nuqs?color=red)](https://www.npmjs.com/package/nuqs)
[![MIT License](https://img.shields.io/github/license/47ng/nuqs.svg?color=blue)](https://github.com/47ng/nuqs/blob/next/LICENSE)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/franky47?color=%23db61a2&label=Sponsors)](https://github.com/sponsors/franky47)
[![CI/CD](https://github.com/47ng/nuqs/actions/workflows/ci-cd.yml/badge.svg?branch=next)](https://github.com/47ng/nuqs/actions/workflows/ci-cd.yml)
[![Depfu](https://badges.depfu.com/badges/acad53fa2b09b1e435a19d6d18f29af4/count.svg)](https://depfu.com/github/47ng/nuqs?project_id=22104)

Type-safe search params state manager for React frameworks. Like `useState`, but stored in the URL query string.

## Features

- 🔀 **new:** Supports Next.js (`app` and `pages` routers), plain React (SPA), Remix, React Router, and custom routers via [adapters](#adapters)
- 🧘‍♀️ Simple: the URL is the source of truth
- 🕰 Replace history or [append](#history) to use the Back button to navigate state updates
- ⚡️ Built-in [parsers](#parsing) for common state types (integer, float, boolean, Date, and more). Create your own parsers for custom types & pretty URLs
- ♊️ Related querystrings with [`useQueryStates`](#usequerystates)
- 📡 [Shallow mode](#shallow) by default for URL query updates, opt-in to notify server components
- 🗃 [Server cache](#accessing-searchparams-in-server-components) for type-safe searchParams access in nested server components
- ⌛️ Support for [`useTransition`](#transitions) to get loading states on server updates

## Documentation

Read the complete documentation at [nuqs.47ng.com](https://nuqs.47ng.com).

## Installation

```shell
pnpm add nuqs
```

```shell
yarn add nuqs
```

```shell
npm install nuqs
```

## Adapters

You will need to wrap your React component tree with an adapter for your framework. _(expand the appropriate section below)_

<details><summary>▲ Next.js (app router)</summary>

> Supported Next.js versions: `>=14.2.0`. For older versions, install `nuqs@^1` (which doesn't need this adapter code).

```tsx
// src/app/layout.tsx
import { NuqsAdapter } from 'nuqs/adapters/next/app'
import { type ReactNode } from 'react'

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html>
      <body>
        <NuqsAdapter>{children}</NuqsAdapter>
      </body>
    </html>
  )
}
```

</details>

<details><summary>▲ Next.js (pages router)</summary>

> Supported Next.js versions: `>=14.2.0`. For older versions, install `nuqs@^1` (which doesn't need this adapter code).

```tsx
// src/pages/_app.tsx
import type { AppProps } from 'next/app'
import { NuqsAdapter } from 'nuqs/adapters/next/pages'

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <NuqsAdapter>
      <Component {...pageProps} />
    </NuqsAdapter>
  )
}
```

</details>

<details><summary>⚛️ Plain React (SPA)</summary>

Example: via Vite or create-react-app.

```tsx
import { NuqsAdapter } from 'nuqs/adapters/react'

createRoot(document.getElementById('root')!).render(
  <NuqsAdapter>
    <App />
  </NuqsAdapter>
)
```

</details>

<details><summary>💿 Remix</summary>

> Supported Remix versions: `@remix-run/react@>=2`

```tsx
// app/root.tsx
import { NuqsAdapter } from 'nuqs/adapters/remix'

// ...

export default function App() {
  return (
    <NuqsAdapter>
      <Outlet />
    </NuqsAdapter>
  )
}
```

</details>

<details><summary><img style="width:1em;height:1em;" src="https://reactrouter.com/_brand/React%20Router%20Brand%20Assets/React%20Router%20Logo/Dark.svg" /> React Router v6
</summary>

> Supported React Router versions: `react-router-dom@^6`

```tsx
import { NuqsAdapter } from 'nuqs/adapters/react-router/v6'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App from './App'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />
  }
])

export function ReactRouter() {
  return (
    <NuqsAdapter>
      <RouterProvider router={router} />
    </NuqsAdapter>
  )
}
```

</details>

<details><summary><img style="width:1em;height:1em;" src="https://reactrouter.com/_brand/React%20Router%20Brand%20Assets/React%20Router%20Logo/Dark.svg" /> React Router v7
</summary>

> Supported React Router versions: `react-router@^7`

```tsx
// app/root.tsx
import { NuqsAdapter } from 'nuqs/adapters/react-router/v7'
import { Outlet } from 'react-router'

// ...

export default function App() {
  return (
    <NuqsAdapter>
      <Outlet />
    </NuqsAdapter>
  )
}
```

</details>

## Usage

```tsx
'use client' // Only works in client components

import { useQueryState } from 'nuqs'

export default () => {
  const [name, setName] = useQueryState('name')
  return (
    <>
      <h1>Hello, {name || 'anonymous visitor'}!</h1>
      <input value={name || ''} onChange={e => setName(e.target.value)} />
      <button onClick={() => setName(null)}>Clear</button>
    </>
  )
}
```

![](https://raw.githubusercontent.com/47ng/nuqs/next/useQueryState.gif)

`useQueryState` takes one required argument: the key to use in the query string.

Like `React.useState`, it returns an array with the value present in the query
string as a string (or `null` if none was found), and a state updater function.

Example outputs for our hello world example:

| URL          | name value | Notes                                                             |
| ------------ | ---------- | ----------------------------------------------------------------- |
| `/`          | `null`     | No `name` key in URL                                              |
| `/?name=`    | `''`       | Empty string                                                      |
| `/?name=foo` | `'foo'`    |
| `/?name=2`   | `'2'`      | Always returns a string by default, see [Parsing](#parsing) below |

## Parsing

If your state type is not a string, you must pass a parsing function in the
second argument object.

We provide parsers for common and more advanced object types:

```ts
import {
  parseAsString,
  parseAsInteger,
  parseAsFloat,
  parseAsBoolean,
  parseAsTimestamp,
  parseAsIsoDateTime,
  parseAsArrayOf,
  parseAsJson,
  parseAsStringEnum,
  parseAsStringLiteral,
  parseAsNumberLiteral
} from 'nuqs'

useQueryState('tag') // defaults to string
useQueryState('count', parseAsInteger)
useQueryState('brightness', parseAsFloat)
useQueryState('darkMode', parseAsBoolean)
useQueryState('after', parseAsTimestamp) // state is a Date
useQueryState('date', parseAsIsoDateTime) // state is a Date
useQueryState('array', parseAsArrayOf(parseAsInteger)) // state is number[]
useQueryState('json', parseAsJson<Point>()) // state is a Point

// Enums (string-based only)
enum Direction {
  up = 'UP',
  down = 'DOWN',
  left = 'LEFT',
  right = 'RIGHT'
}

const [direction, setDirection] = useQueryState(
  'direction',
  parseAsStringEnum<Direction>(Object.values(Direction)) // pass a list of allowed values
    .withDefault(Direction.up)
)

// Literals (string-based only)
const colors = ['red', 'green', 'blue'] as const

const [color, setColor] = useQueryState(
  'color',
  parseAsStringLiteral(colors) // pass a readonly list of allowed values
    .withDefault('red')
)

// Literals (number-based only)
const diceSides = [1, 2, 3, 4, 5, 6] as const

const [side, setSide] = useQueryState(
  'side',
  parseAsNumberLiteral(diceSides) // pass a readonly list of allowed values
    .withDefault(4)
)
```

You may pass a custom set of `parse` and `serialize` functions:

```tsx
import { useQueryState } from 'nuqs'

export default () => {
  const [hex, setHex] = useQueryState('hex', {
    // TypeScript will automatically infer it's a number
    // based on what `parse` returns.
    parse: (query: string) => parseInt(query, 16),
    serialize: value => value.toString(16)
  })
}
```

### Using parsers in Server Components

> Note: see the [Accessing searchParams in server components](#accessing-searchparams-in-server-components)
> section for a more user-friendly way to achieve type-safety.

If you wish to parse the searchParams in server components, you'll need to
import the parsers from `nuqs/server`, which doesn't include
the `"use client"` directive.

You can then use the `parseServerSide` method:

```tsx
import { parseAsInteger } from 'nuqs/server'

type PageProps = {
  searchParams: {
    counter?: string | string[]
  }
}

const counterParser = parseAsInteger.withDefault(1)

export default function ServerPage({ searchParams }: PageProps) {
  const counter = counterParser.parseServerSide(searchParams.counter)
  console.log('Server side counter: %d', counter)
  return (
    ...
  )
}
```

See the [server-side parsing demo](<./packages/docs/src/app/playground/(demos)/pagination>)
for a live example showing how to reuse parser configurations between
client and server code.

> Note: parsers **don't validate** your data. If you expect positive integers
> or JSON-encoded objects of a particular shape, you'll need to feed the result
> of the parser to a schema validation library, like [Zod](https://zod.dev).

## Default value

When the query string is not present in the URL, the default behaviour is to
return `null` as state.

It can make state updating and UI rendering tedious. Take this example of a simple counter stored in the URL:

```tsx
import { useQueryState, parseAsInteger } from 'nuqs'

export default () => {
  const [count, setCount] = useQueryState('count', parseAsInteger)
  return (
    <>
      <pre>count: {count}</pre>
      <button onClick={() => setCount(0)}>Reset</button>
      {/* handling null values in setCount is annoying: */}
      <button onClick={() => setCount(c => c ?? 0 + 1)}>+</button>
      <button onClick={() => setCount(c => c ?? 0 - 1)}>-</button>
      <button onClick={() => setCount(null)}>Clear</button>
    </>
  )
}
```

You can specify a default value to be returned in this case:

```ts
const [count, setCount] = useQueryState('count', parseAsInteger.withDefault(0))

const increment = () => setCount(c => c + 1) // c will never be null
const decrement = () => setCount(c => c - 1) // c will never be null
const clearCount = () => setCount(null) // Remove query from the URL
```

Note: the default value is internal to React, it will **not** be written to the
URL.

Setting the state to `null` will remove the key in the query string and set the
state to the default value.

## Options

### History

By default, state updates are done by replacing the current history entry with
the updated query when state changes.

You can see this as a sort of `git squash`, where all state-changing
operations are merged into a single history value.

You can also opt-in to push a new history item for each state change,
per key, which will let you use the Back button to navigate state
updates:

```ts
// Default: replace current history with new state
useQueryState('foo', { history: 'replace' })

// Append state changes to history:
useQueryState('foo', { history: 'push' })
```

Any other value for the `history` option will fallback to the default.

You can also override the history mode when calling the state updater function:

```ts
const [query, setQuery] = useQueryState('q', { history: 'push' })

// This overrides the hook declaration setting:
setQuery(null, { history: 'replace' })
```

### Shallow

> Note: this feature only applies to Next.js

By default, query state updates are done in a _client-first_ manner: there are
no network calls to the server.

This is equivalent to the `shallow` option of the Next.js pages router set to `true`,
or going through the experimental [`windowHistorySupport`](https://github.com/vercel/next.js/discussions/48110)
flag in the app router.

To opt-in to query updates notifying the server (to re-run `getServerSideProps`
in the pages router and re-render Server Components on the app router),
you can set `shallow` to `false`:

```ts
const [state, setState] = useQueryState('foo', { shallow: false })

// You can also pass the option on calls to setState:
setState('bar', { shallow: false })
```

### Throttling URL updates

Because of browsers rate-limiting the History API, internal updates to the
URL are queued and throttled to a default of 50ms, which seems to satisfy
most browsers even when sending high-frequency query updates, like binding
to a text input or a slider.

Safari's rate limits are much higher and would require a throttle of around 340ms.
If you end up needing a longer time between updates, you can specify it in the
options:

```ts
useQueryState('foo', {
  // Send updates to the server maximum once every second
  shallow: false,
  throttleMs: 1000
})

// You can also pass the option on calls to setState:
setState('bar', { throttleMs: 1000 })
```

> Note: the state returned by the hook is always updated instantly, to keep UI responsive.
> Only changes to the URL, and server requests when using `shallow: false`, are throttled.

If multiple hooks set different throttle values on the same event loop tick,
the highest value will be used. Also, values lower than 50ms will be ignored,
to avoid rate-limiting issues. [Read more](https://francoisbest.com/posts/2023/storing-react-state-in-the-url-with-nextjs#batching--throttling).

### Transitions

When combined with `shallow: false`, you can use the `useTransition` hook to get
loading states while the server is re-rendering server components with the
updated URL.

Pass in the `startTransition` function from `useTransition` to the options
to enable this behaviour:

```tsx
'use client'

import React from 'react'
import { useQueryState, parseAsString } from 'nuqs'

function ClientComponent({ data }) {
  // 1. Provide your own useTransition hook:
  const [isLoading, startTransition] = React.useTransition()
  const [query, setQuery] = useQueryState(
    'query',
    // 2. Pass the `startTransition` as an option:
    parseAsString().withOptions({
      startTransition,
      shallow: false // opt-in to notify the server (Next.js only)
    })
  )
  // 3. `isLoading` will be true while the server is re-rendering
  // and streaming RSC payloads, when the query is updated via `setQuery`.

  // Indicate loading state
  if (isLoading) return <div>Loading...</div>

  // Normal rendering with data
  return <div>{/*...*/}</div>
}
```

## Configuring parsers, default value & options

You can use a builder pattern to facilitate specifying all of those things:

```ts
useQueryState(
  'counter',
  parseAsInteger.withDefault(0).withOptions({
    history: 'push',
    shallow: false
  })
)
```

You can get this pattern for your custom parsers too, and compose them
with others:

```ts
import { createParser, parseAsHex } from 'nuqs'

// Wrapping your parser/serializer in `createParser`
// gives it access to the builder pattern & server-side
// parsing capabilities:
const hexColorSchema = createParser({
  parse(query) {
    if (query.length !== 6) {
      return null // always return null for invalid inputs
    }
    return {
      // When composing other parsers, they may return null too.
      r: parseAsHex.parse(query.slice(0, 2)) ?? 0x00,
      g: parseAsHex.parse(query.slice(2, 4)) ?? 0x00,
      b: parseAsHex.parse(query.slice(4)) ?? 0x00
    }
  },
  serialize({ r, g, b }) {
    return (
      parseAsHex.serialize(r) +
      parseAsHex.serialize(g) +
      parseAsHex.serialize(b)
    )
  }
})
  // Eg: set common options directly
  .withOptions({ history: 'push' })

// Or on usage:
useQueryState(
  'tribute',
  hexColorSchema.withDefault({
    r: 0x66,
    g: 0x33,
    b: 0x99
  })
)
```

Note: see this example running in the [hex-colors demo](<./packages/docs/src/app/playground/(demos)/hex-colors/page.tsx>).

## Multiple Queries (batching)

You can call as many state update function as needed in a single event loop
tick, and they will be applied to the URL asynchronously:

```ts
const MultipleQueriesDemo = () => {
  const [lat, setLat] = useQueryState('lat', parseAsFloat)
  const [lng, setLng] = useQueryState('lng', parseAsFloat)
  const randomCoordinates = React.useCallback(() => {
    setLat(Math.random() * 180 - 90)
    setLng(Math.random() * 360 - 180)
  }, [])
}
```

If you wish to know when the URL has been updated, and what it contains, you can
await the Promise returned by the state updater function, which gives you the
updated URLSearchParameters object:

```ts
const randomCoordinates = React.useCallback(() => {
  setLat(42)
  return setLng(12)
}, [])

randomCoordinates().then((search: URLSearchParams) => {
  search.get('lat') // 42
  search.get('lng') // 12, has been queued and batch-updated
})
```

<details>
<summary><em>Implementation details (Promise caching)</em></summary>

The returned Promise is cached until the next flush to the URL occurs,
so all calls to a setState (of any hook) in the same event loop tick will
return the same Promise reference.

Due to throttling of calls to the Web History API, the Promise may be cached
for several ticks. Batched updates will be merged and flushed once to the URL.
This means not every setState will reflect to the URL, if another one comes
overriding it before flush occurs.

The returned React state will reflect all set values instantly,
to keep UI responsive.

---

</details>

## `useQueryStates`

For query keys that should always move together, you can use `useQueryStates`
with an object containing each key's type:

```ts
import { useQueryStates, parseAsFloat } from 'nuqs'

const [coordinates, setCoordinates] = useQueryStates(
  {
    lat: parseAsFloat.withDefault(45.18),
    lng: parseAsFloat.withDefault(5.72)
  },
  {
    history: 'push'
  }
)

const { lat, lng } = coordinates

// Set all (or a subset of) the keys in one go:
const search = await setCoordinates({
  lat: Math.random() * 180 - 90,
  lng: Math.random() * 360 - 180
})
```

## Loaders

To parse search params as a one-off operation, you can use a **loader function**:

```tsx
import { createLoader } from 'nuqs' // or 'nuqs/server'

const searchParams = {
  q: parseAsString,
  page: parseAsInteger.withDefault(1)
}

const loadSearchParams = createLoader(searchParams)

const { q, page } = loadSearchParams('?q=hello&page=2')
```

It accepts various types of inputs (strings, URL, URLSearchParams, Request, Promises, etc.). [Read more](https://nuqs.47ng.com/docs/server-side#loaders)

## Accessing searchParams in Server Components

If you wish to access the searchParams in a deeply nested Server Component
(ie: not in the Page component), you can use `createSearchParamsCache`
to do so in a type-safe manner.

> Note: parsers **don't validate** your data. If you expect positive integers
> or JSON-encoded objects of a particular shape, you'll need to feed the result
> of the parser to a schema validation library, like [Zod](https://zod.dev).

```tsx
// searchParams.ts
import {
  createSearchParamsCache,
  parseAsInteger,
  parseAsString
} from 'nuqs/server'
// Note: import from 'nuqs/server' to avoid the "use client" directive

export const searchParamsCache = createSearchParamsCache({
  // List your search param keys and associated parsers here:
  q: parseAsString.withDefault(''),
  maxResults: parseAsInteger.withDefault(10)
})

// page.tsx
import { searchParamsCache } from './searchParams'

export default function Page({
  searchParams
}: {
  searchParams: Record<string, string | string[] | undefined>
}) {
  // ⚠️ Don't forget to call `parse` here.
  // You can access type-safe values from the returned object:
  const { q: query } = searchParamsCache.parse(searchParams)
  return (
    <div>
      <h1>Search Results for {query}</h1>
      <Results />
    </div>
  )
}

function Results() {
  // Access type-safe search params in children server components:
  const maxResults = searchParamsCache.get('maxResults')
  return <span>Showing up to {maxResults} results</span>
}
```

The cache will only be valid for the current page render
(see React's [`cache`](https://react.dev/reference/react/cache) function).

Note: the cache only works for **server components**, but you may share your
parser declaration with `useQueryStates` for type-safety in client components:

```tsx
// searchParams.ts
import { parseAsFloat, createSearchParamsCache } from 'nuqs/server'

export const coordinatesParsers = {
  lat: parseAsFloat.withDefault(45.18),
  lng: parseAsFloat.withDefault(5.72)
}
export const coordinatesCache = createSearchParamsCache(coordinatesParsers)

// page.tsx
import { coordinatesCache } from './searchParams'
import { Server } from './server'
import { Client } from './client'

export default async function Page({ searchParams }) {
  await coordinatesCache.parse(searchParams)
  return (
    <>
      <Server />
      <Suspense>
        <Client />
      </Suspense>
    </>
  )
}

// server.tsx
import { coordinatesCache } from './searchParams'

export function Server() {
  const { lat, lng } = coordinatesCache.all()
  // or access keys individually:
  const lat = coordinatesCache.get('lat')
  const lng = coordinatesCache.get('lng')
  return (
    <span>
      Latitude: {lat} - Longitude: {lng}
    </span>
  )
}

// client.tsx
// prettier-ignore
;'use client'

import { useQueryStates } from 'nuqs'
import { coordinatesParsers } from './searchParams'

export function Client() {
  const [{ lat, lng }, setCoordinates] = useQueryStates(coordinatesParsers)
  // ...
}
```

## Serializer helper

To populate `<Link>` components with state values, you can use the `createSerializer`
helper.

Pass it an object describing your search params, and it will give you a function
to call with values, that generates a query string serialized as the hooks would do.

Example:

```ts
import {
  createSerializer,
  parseAsInteger,
  parseAsIsoDateTime,
  parseAsString,
  parseAsStringLiteral
} from 'nuqs/server'

const searchParams = {
  search: parseAsString,
  limit: parseAsInteger,
  from: parseAsIsoDateTime,
  to: parseAsIsoDateTime,
  sortBy: parseAsStringLiteral(['asc', 'desc'] as const)
}

// Create a serializer function by passing the description of the search params to accept
const serialize = createSerializer(searchParams)

// Then later, pass it some values (a subset) and render them to a query string
serialize({
  search: 'foo bar',
  limit: 10,
  from: new Date('2024-01-01'),
  // here, we omit `to`, which won't be added
  sortBy: null // null values are also not rendered
})
// ?search=foo+bar&limit=10&from=2024-01-01T00:00:00.000Z
```

### Base parameter

The returned `serialize` function can take a base parameter over which to
append/amend the search params:

```ts
serialize('/path?baz=qux', { foo: 'bar' }) // /path?baz=qux&foo=bar

const search = new URLSearchParams('?baz=qux')
serialize(search, { foo: 'bar' }) // ?baz=qux&foo=bar

const url = new URL('https://example.com/path?baz=qux')
serialize(url, { foo: 'bar' }) // https://example.com/path?baz=qux&foo=bar

// Passing null removes existing values
serialize('?remove=me', { foo: 'bar', remove: null }) // ?foo=bar
```

## Parser type inference

To access the underlying type returned by a parser, you can use the
`inferParserType` type helper:

```ts
import { parseAsInteger, type inferParserType } from 'nuqs' // or 'nuqs/server'

const intNullable = parseAsInteger
const intNonNull = parseAsInteger.withDefault(0)

inferParserType<typeof intNullable> // number | null
inferParserType<typeof intNonNull> // number
```

For an object describing parsers (that you'd pass to `createSearchParamsCache`
or to `useQueryStates`, `inferParserType` will
return the type of the object with the parsers replaced by their inferred types:

```ts
import { parseAsBoolean, parseAsInteger, type inferParserType } from 'nuqs' // or 'nuqs/server'

const parsers = {
  a: parseAsInteger,
  b: parseAsBoolean.withDefault(false)
}

inferParserType<typeof parsers>
// { a: number | null, b: boolean }
```

## Testing

Since nuqs v2, you can use a testing adapter to unit-test components using
`useQueryState` and `useQueryStates` in isolation, without needing to mock
your framework or router.

Here's an example using Testing Library and Vitest:

```tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { NuqsTestingAdapter, type UrlUpdateEvent } from 'nuqs/adapters/testing'
import { describe, expect, it, vi } from 'vitest'
import { CounterButton } from './counter-button'

it('should increment the count when clicked', async () => {
  const user = userEvent.setup()
  const onUrlUpdate = vi.fn<[UrlUpdateEvent]>()
  render(<CounterButton />, {
    // Setup the test by passing initial search params / querystring,
    // and give it a function to call on URL updates
    wrapper: ({ children }) => (
      <NuqsTestingAdapter searchParams="?count=42" onUrlUpdate={onUrlUpdate}>
        {children}
      </NuqsTestingAdapter>
    )
  })
  // Initial state assertions: there's a clickable button displaying the count
  const button = screen.getByRole('button')
  expect(button).toHaveTextContent('count is 42')
  // Act
  await user.click(button)
  // Assert changes in the state and in the (mocked) URL
  expect(button).toHaveTextContent('count is 43')
  expect(onUrlUpdate).toHaveBeenCalledOnce()
  expect(onUrlUpdate.mock.calls[0][0].queryString).toBe('?count=43')
  expect(onUrlUpdate.mock.calls[0][0].searchParams.get('count')).toBe('43')
  expect(onUrlUpdate.mock.calls[0][0].options.history).toBe('push')
})
```

See [#259](https://github.com/47ng/nuqs/issues/259) for more testing-related discussions.

## Debugging

You can enable debug logs in the browser by setting the `debug` item in localStorage
to `nuqs`, and reload the page.

```js
// In your devtools:
localStorage.setItem('debug', 'nuqs')
```

> Note: unlike the `debug` package, this will not work with wildcards, but
> you can combine it: `localStorage.setItem('debug', '*,nuqs')`

Log lines will be prefixed with `[nuqs]` for `useQueryState` and `[nuq+]` for
`useQueryStates`, along with other internal debug logs.

User timings markers are also recorded, for advanced performance analysis using
your browser's devtools.

Providing debug logs when opening an [issue](https://github.com/47ng/nuqs/issues)
is always appreciated. 🙏

### SEO

If your page uses query strings for local-only state, you should add a
canonical URL to your page, to tell SEO crawlers to ignore the query string
and index the page without it.

In the app router, this is done via the metadata object:

```ts
import type { Metadata } from 'next'

export const metadata: Metadata = {
  alternates: {
    canonical: '/url/path/without/querystring'
  }
}
```

If however the query string is defining what content the page is displaying
(eg: YouTube's watch URLs, like `https://www.youtube.com/watch?v=dQw4w9WgXcQ`),
your canonical URL should contain relevant query strings, and you can still
use `useQueryState` to read it:

```ts
// page.tsx
import type { Metadata, ResolvingMetadata } from 'next'
import { useQueryState } from 'nuqs'
import { parseAsString } from 'nuqs/server'

type Props = {
  searchParams: { [key: string]: string | string[] | undefined }
}

export async function generateMetadata({
  searchParams
}: Props): Promise<Metadata> {
  const videoId = parseAsString.parseServerSide(searchParams.v)
  return {
    alternates: {
      canonical: `/watch?v=${videoId}`
    }
  }
}
```

### Lossy serialization

If your serializer loses precision or doesn't accurately represent
the underlying state value, you will lose this precision when
reloading the page or restoring state from the URL (eg: on navigation).

Example:

```ts
const geoCoordParser = {
  parse: parseFloat,
  serialize: v => v.toFixed(4) // Loses precision
}

const [lat, setLat] = useQueryState('lat', geoCoordParser)
```

Here, setting a latitude of 1.23456789 will render a URL query string
of `lat=1.2345`, while the internal `lat` state will be correctly
set to 1.23456789.

Upon reloading the page, the state will be incorrectly set to 1.2345.

## License

[MIT](https://github.com/47ng/nuqs/blob/next/LICENSE)

Made with ❤️ by [François Best](https://francoisbest.com)

Using this package at work ? [Sponsor me](https://github.com/sponsors/franky47)
to help with support and maintenance.

![Project analytics and stats](https://repobeats.axiom.co/api/embed/3ee740e4729dce3992bfa8c74645cfebad8ba034.svg 'Repobeats analytics image')
