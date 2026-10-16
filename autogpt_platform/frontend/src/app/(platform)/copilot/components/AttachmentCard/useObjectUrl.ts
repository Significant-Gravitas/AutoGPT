import { useEffect, useState } from "react";

/** A `blob:` URL for a local file, revoked when the file changes or the
 *  caller unmounts. Undefined until the first effect runs, or with no file. */
export function useObjectUrl(file?: Blob) {
  const [url, setUrl] = useState<string>();

  useEffect(() => {
    if (!file) return;
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => {
      URL.revokeObjectURL(next);
      setUrl(undefined);
    };
  }, [file]);

  return url;
}
