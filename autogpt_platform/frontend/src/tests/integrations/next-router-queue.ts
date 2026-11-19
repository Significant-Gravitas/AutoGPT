import { createRequire, registerHooks } from "node:module";

// Next's client router queue as installed (node_modules/next, with
// patches/next@15.5.28.patch applied), driven by hand. In the browser every
// server action and navigation goes through this queue; a test sends its
// fake server actions and navigations through it the same way, and decides
// when each one finishes.

const nodeRequire = createRequire(import.meta.url);
const QUEUE_MODULE = "next/dist/client/components/app-router-instance";
const TYPES_MODULE =
  "next/dist/client/components/router-reducer/router-reducer-types";

interface QueuedPayload {
  type: string;
  run?: () => Promise<unknown>;
  routerFails?: boolean;
  arrived?: Promise<void>;
  resolve?: (value: unknown) => void;
  reject?: (reason: unknown) => void;
}

interface ServerActionOptions {
  /**
   * The router can't apply the server's answer, so its reducer rejects too,
   * as Next's does when a refetch it needs for the answer fails.
   */
  routerFails?: boolean;
}

interface ActionQueue {
  action: (state: unknown, payload: QueuedPayload) => Promise<unknown>;
  dispatch: (payload: QueuedPayload, setState: (next: unknown) => void) => void;
}

let hooks: { deregister: () => void } | null = null;

export function createRouterQueue() {
  prepareNodeForNextClientModules();
  // One queue per page in Next, so a fresh module gives each test its own.
  delete nodeRequire.cache[nodeRequire.resolve(QUEUE_MODULE)];
  const { createMutableActionQueue } = nodeRequire(QUEUE_MODULE) as {
    createMutableActionQueue: (state: unknown, hooks: null) => ActionQueue;
  };
  const { ACTION_NAVIGATE, ACTION_SERVER_ACTION } = nodeRequire(
    TYPES_MODULE,
  ) as Record<string, string>;
  const state = {};
  const queue = createMutableActionQueue(state, null);
  // Stands in for Next's reducers. As the server action reducer does, the
  // caller gets the action's answer before the router state moves on.
  queue.action = async (_state, payload) => {
    if (payload.type === ACTION_SERVER_ACTION) {
      try {
        payload.resolve?.(await payload.run?.());
      } catch (error) {
        payload.reject?.(error);
        if (payload.routerFails) throw error;
      }
    } else if (payload.type === ACTION_NAVIGATE) {
      await payload.arrived;
    }
    return state;
  };
  function noop() {}
  // React takes the router state as a promise; a failed action rejects it.
  function setState(next: unknown) {
    if (next instanceof Promise) next.catch(noop);
  }

  return {
    /** Sends a server action, as `callServer` does; `run` is the server. */
    callServer<T>(run: () => Promise<T>, options: ServerActionOptions = {}) {
      return new Promise<T>((resolve, reject) => {
        queue.dispatch(
          {
            type: ACTION_SERVER_ACTION,
            run,
            routerFails: options.routerFails,
            resolve: resolve as (value: unknown) => void,
            reject,
          },
          setState,
        );
      });
    },
    /** Starts a client-side navigation that lasts until `arrive()`. */
    navigate() {
      let arrive = noop;
      const arrived = new Promise<void>((resolve) => {
        arrive = resolve;
      });
      queue.dispatch({ type: ACTION_NAVIGATE, arrived }, setState);
      return { arrive };
    },
  };
}

export function disposeRouterQueue() {
  hooks?.deregister();
  hooks = null;
  Reflect.deleteProperty(globalThis, "__webpack_require__");
}

// Next's bundler maps this import to its compiled copy, and the browser
// bundle provides webpack's runtime; plain Node has neither.
function prepareNodeForNextClientModules() {
  hooks ??= registerHooks({
    resolve(specifier, context, nextResolve) {
      return nextResolve(
        specifier === "react-server-dom-webpack/client"
          ? "next/dist/compiled/react-server-dom-webpack/client"
          : specifier,
        context,
      );
    },
  });
  Object.assign(globalThis, {
    __webpack_require__: Object.assign(() => undefined, { u: () => "" }),
  });
}
