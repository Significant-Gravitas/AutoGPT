/*
 * Dev-only React render counter. Loaded by src/app/layout.tsx only when
 * NEXT_PUBLIC_RENDER_COUNTER=1, before React, so React reports every commit
 * to __REACT_DEVTOOLS_GLOBAL_HOOK__. Plain JS, never bundled.
 *
 * Console: __renderCounter.print() / .report() / .reset() / .start() / .stop()
 * Docs: docs/engineering/frontend-perf-measurement.md
 */
(function () {
  "use strict";
  if (typeof window === "undefined" || window.__renderCounter) return;

  // React 18 fiber tags and flags.
  var FUNCTION = 0;
  var CLASS = 1;
  var FORWARD_REF = 11;
  var SIMPLE_MEMO = 15;
  var PERFORMED_WORK = 1;
  // MemoComponent (14) is left out on purpose: React marks both the memo
  // wrapper and the component inside it, which would count one render twice.
  var COUNTED_TAGS = [FUNCTION, CLASS, FORWARD_REF, SIMPLE_MEMO];

  var recording = true;
  var commits = 0;
  var totalRenders = 0;
  var maxRendersPerCommit = 0;
  var counts = new Map();
  var causes = new Map();

  function bump(map, key) {
    map.set(key, (map.get(key) || 0) + 1);
  }

  function nameOf(type) {
    if (!type) return "Anonymous";
    var name =
      type.displayName ||
      type.name ||
      (type.type && (type.type.displayName || type.type.name)) ||
      (type.render && (type.render.displayName || type.render.name)) ||
      "Anonymous";
    var match;
    while ((match = /^(?:Memo|ForwardRef)\((.*)\)$/.exec(name))) {
      name = match[1];
    }
    return name || "Anonymous";
  }

  function changedStateSlot(fiber) {
    var before = fiber.alternate.memoizedState;
    var after = fiber.memoizedState;
    if (fiber.tag === CLASS) return before !== after ? "state" : null;
    var index = 0;
    while (before && after && typeof after === "object" && "next" in after) {
      if (after.queue !== null && !Object.is(before.memoizedState, after.memoizedState)) {
        return "state#" + index;
      }
      before = before.next;
      after = after.next;
      index++;
    }
    return null;
  }

  function changedPropKeys(before, after) {
    var keys = [];
    var seen = {};
    [before, after].forEach(function (props) {
      if (!props || typeof props !== "object") return;
      Object.keys(props).forEach(function (key) {
        if (seen[key]) return;
        seen[key] = true;
        if (!Object.is(before && before[key], after && after[key])) keys.push(key);
      });
    });
    return keys;
  }

  function causeOf(fiber, parentName) {
    if (fiber.alternate === null) return "mount";
    var state = changedStateSlot(fiber);
    if (state) return state;
    var before = fiber.alternate.memoizedProps;
    var after = fiber.memoizedProps;
    if (before !== after) {
      var keys = changedPropKeys(before, after);
      if (keys.length === 0) return "parent-same-props<-" + parentName;
      var more = keys.length > 4 ? ",+" + (keys.length - 4) : "";
      return "props:" + keys.slice(0, 4).join(",") + more;
    }
    return "context";
  }

  function walk(root) {
    var rendersThisCommit = 0;
    var stack = [[root.current, "Root"]];
    while (stack.length) {
      var item = stack.pop();
      var fiber = item[0];
      var parentName = item[1];
      var rendered =
        COUNTED_TAGS.indexOf(fiber.tag) !== -1 &&
        (fiber.flags & PERFORMED_WORK) !== 0;
      var childParent = parentName;
      if (rendered) {
        var name = nameOf(fiber.type);
        rendersThisCommit++;
        bump(counts, name);
        bump(causes, name + " " + causeOf(fiber, parentName));
        childParent = name;
      } else if (COUNTED_TAGS.indexOf(fiber.tag) !== -1) {
        childParent = nameOf(fiber.type);
      }
      if (fiber.sibling) stack.push([fiber.sibling, parentName]);
      // A bailed-out subtree keeps the fibers (and stale flags) of an older
      // commit, so nothing below it rendered in this one.
      var bailedOut =
        fiber.alternate !== null && fiber.child === fiber.alternate.child && !rendered;
      if (fiber.child && !bailedOut) stack.push([fiber.child, childParent]);
    }
    return rendersThisCommit;
  }

  function onCommit(root) {
    if (!recording || !root || !root.current) return;
    try {
      var renders = walk(root);
      commits++;
      totalRenders += renders;
      if (renders > maxRendersPerCommit) maxRendersPerCommit = renders;
    } catch (error) {
      console.warn("[render-counter] walk failed", error);
    }
  }

  function sorted(map) {
    return Array.from(map, function (entry) {
      return { key: entry[0], count: entry[1] };
    }).sort(function (a, b) {
      return b.count - a.count;
    });
  }

  function installHook() {
    var hook = window.__REACT_DEVTOOLS_GLOBAL_HOOK__;
    if (!hook) {
      var nextId = 1;
      var noop = function () {};
      hook = {
        supportsFiber: true,
        renderers: new Map(),
        inject: function (renderer) {
          var id = nextId++;
          hook.renderers.set(id, renderer);
          return id;
        },
        onCommitFiberRoot: noop,
        onCommitFiberUnmount: noop,
        onPostCommitFiberRoot: noop,
        onScheduleFiberRoot: noop,
        checkDCE: noop,
        isDisabled: false,
        on: noop,
        off: noop,
        emit: noop,
        sub: function () {
          return noop;
        },
      };
      window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = hook;
    }
    var previous = hook.onCommitFiberRoot;
    hook.onCommitFiberRoot = function (rendererID, root) {
      onCommit(root);
      if (typeof previous === "function") return previous.apply(this, arguments);
    };
  }

  window.__renderCounter = {
    start: function () {
      recording = true;
    },
    stop: function () {
      recording = false;
    },
    reset: function () {
      commits = 0;
      totalRenders = 0;
      maxRendersPerCommit = 0;
      counts.clear();
      causes.clear();
    },
    report: function () {
      return {
        commits: commits,
        totalRenders: totalRenders,
        rendersPerCommit: commits ? totalRenders / commits : 0,
        maxRendersPerCommit: maxRendersPerCommit,
        components: sorted(counts),
        causes: sorted(causes),
      };
    },
    print: function (limit) {
      var report = this.report();
      var top = limit || 20;
      console.log(
        "[render-counter] " + report.commits + " commits, " + report.totalRenders + " renders"
      );
      console.table(report.components.slice(0, top));
      console.table(report.causes.slice(0, top));
      return report;
    },
  };

  installHook();
})();
