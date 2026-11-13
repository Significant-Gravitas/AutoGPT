import { useEffect, useRef, useState } from "react";

// "prompt" after a refusal means the prompt was closed rather than answered,
// so asking again shows it again. "unknown" is a browser that cannot say:
// no Permissions API (older Safari) or no "microphone" name in it (Firefox).
export type MicrophonePermission = "prompt" | "granted" | "denied" | "unknown";

interface Args {
  // Only for a change the browser reports, e.g. the user allowing the site
  // from the address bar. Not called for the first reading.
  onChange?: (state: MicrophonePermission) => void;
}

export function useMicrophonePermission({ onChange }: Args = {}) {
  const [state, setState] = useState<MicrophonePermission>("unknown");
  const onChangeRef = useRef(onChange);

  useEffect(() => {
    onChangeRef.current = onChange;
  });

  useEffect(() => {
    let status: PermissionStatus | null = null;
    let isUnmounted = false;

    function handleChange() {
      if (!status) return;
      const next = toMicrophonePermission(status);
      setState(next);
      onChangeRef.current?.(next);
    }

    void queryMicrophone().then((answer) => {
      if (isUnmounted || !answer) return;
      status = answer;
      setState(toMicrophonePermission(answer));
      answer.addEventListener("change", handleChange);
    });

    return () => {
      isUnmounted = true;
      status?.removeEventListener("change", handleChange);
    };
  }, []);

  // A dismissed prompt fires no `change`, so after a refusal the state is
  // read again rather than trusted.
  async function refresh() {
    const next = toMicrophonePermission(await queryMicrophone());
    setState(next);
    return next;
  }

  return { state, refresh };
}

async function queryMicrophone() {
  try {
    const status = await navigator.permissions?.query({
      name: "microphone" as PermissionName,
    });
    return status ?? null;
  } catch {
    return null;
  }
}

function toMicrophonePermission(
  status: PermissionStatus | null,
): MicrophonePermission {
  const state = status?.state;
  return state === "prompt" || state === "granted" || state === "denied"
    ? state
    : "unknown";
}
