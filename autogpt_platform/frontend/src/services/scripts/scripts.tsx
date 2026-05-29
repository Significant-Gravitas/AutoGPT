// Generic client-side script loader service

declare global {
  interface Window {
    __scriptLoaderCache?: Map<string, Promise<void>>;
  }
}

export type LoadScriptOptions = {
  async?: boolean;
  defer?: boolean;
  attrs?: Record<string, string>;
  crossOrigin?: string;
  referrerPolicy?: string;
};

export function loadScript(src: string, options?: LoadScriptOptions) {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Cannot load scripts on server"));
  }

  if (!window.__scriptLoaderCache) {
    window.__scriptLoaderCache = new Map();
  }

  const cache = window.__scriptLoaderCache;
  const cached = cache.get(src);

  if (cached) {
    return cached;
  }

  const promise = new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${src}"]`,
    );

    if (existing && (existing as any).__loaded) {
      return resolve();
    }

    const script = existing || document.createElement("script");

    if (!existing) {
      document.head.appendChild(script);
    }

    script.src = src;
    script.async = options?.async ?? true;
    script.defer = options?.defer ?? true;

    if (options?.crossOrigin) {
      script.crossOrigin = options.crossOrigin;
    }

    if (options?.referrerPolicy) {
      script.referrerPolicy = options.referrerPolicy as any;
    }

    if (options?.attrs) {
      Object.entries(options.attrs).forEach(([k, v]) =>
        script.setAttribute(k, v),
      );
    }

    script.onload = () => {
      (script as any).__loaded = true;
      resolve();
    };

    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
  });

  cache.set(src, promise);

  return promise;
}
