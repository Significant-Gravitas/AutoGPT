"use client";

import { FILE_TYPE_KEYS, FileIllustration } from "./FileIllustration";

export function FileTypeMarquee() {
  const items = [...FILE_TYPE_KEYS, ...FILE_TYPE_KEYS];

  return (
    <div
      className="pointer-events-none relative mx-auto w-full max-w-lg overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_15%,black_85%,transparent)]"
      data-testid="artifacts-file-type-marquee"
    >
      <div className="flex w-max animate-marquee-x items-end gap-4 py-2">
        {items.map((key, i) => (
          <div key={`${key}-${i}`} className="px-4 py-3">
            <FileIllustration typeKey={key} />
          </div>
        ))}
      </div>
    </div>
  );
}
