import React from "react";
import Button from "./Button";

import { useTranslation } from "next-i18next";

export default function Dialog({
  header,
  children,
  isShown,
  close,
  footerButton,
}: {
  header: React.ReactNode;
  children: React.ReactNode;
  isShown: boolean;
  close: () => void;
  footerButton?: React.ReactNode;
}) {
  const [ t ] = useTranslation()
  if (!isShown) {
    return <>{null}</>;
  }

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center overflow-y-auto overflow-x-hidden bg-black/70 p-3 font-mono text-white outline-none transition-all">
      <div
        className="absolute bottom-0 left-0 right-0 top-0 "
        onClick={close}
      />
      <div className="relative mx-auto my-6 w-auto max-w-3xl rounded-lg border-2 border-zinc-600">
        {/*content*/}
        <div
          className="relative z-50 flex w-full flex-col rounded-lg border-0 bg-[#3a3a3a] shadow-lg outline-none focus:outline-none"
          onClick={(e) => e.stopPropagation()} // Avoid closing the modal
        >
          {/*header*/}
          <div className="flex items-start justify-between rounded-t border-b-2 border-solid border-white/20 p-5">
            <h3 className="font-mono text-3xl font-semibold">{header}</h3>
            <button className="float-right ml-auto border-0 bg-transparent p-1 text-3xl font-semibold leading-none opacity-5 outline-none focus:outline-none">
              <span className="block h-6 w-6 bg-transparent text-2xl opacity-5 outline-none focus:outline-none">
                ×
              </span>
            </button>
          </div>
          {/*body*/}
          <div className="text-md relative my-3 max-h-[50vh] flex-auto overflow-y-auto p-3 leading-relaxed">
            {children}
          </div>
          {/*footer*/}
          <div className="flex items-center justify-end gap-2 rounded-b border-t-2 border-solid border-white/20 p-2">
            <Button
              enabledClassName="bg-yellow-600 hover:bg-yellow-500"
              onClick={close}
            >
              {t('Close')}
            </Button>
            {footerButton}
          </div>
        </div>
      </div>
    </div>
  );
}
