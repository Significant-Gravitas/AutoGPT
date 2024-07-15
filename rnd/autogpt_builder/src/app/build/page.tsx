"use client";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import FlowEditor from '@/components/Flow';

export default function Home() {
  return (
      <div className="flex flex-col items-center px-12">
          <div className="z-10 w-full items-center justify-between font-mono text-sm lg:flex">
              <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-600 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-900 dark:bg-zinc-900 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
                  Get started by adding a&nbsp;
                  <code className="font-mono font-bold">node</code>
              </p>
              <div
                  className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:size-auto lg:bg-none"
              >
                  <a
                      className="pointer-events-none flex place-items-center gap-2 p-8 lg:pointer-events-auto lg:p-0"
                      href="https://news.agpt.co/"
                      target="_blank"
                      rel="noopener noreferrer"
                  >
                      By{" "}
                      <Image
                          src="/AUTOgpt_Logo_dark.png"
                          alt="AutoGPT Logo"
                          width={100}
                          height={24}
                          priority
                      />
                  </a>
              </div>
          </div>

          <div className="w-full flex justify-center mt-10">
              <FlowEditor
                className="flow-container w-full min-h-[75vh] border border-gray-300 dark:border-gray-700 rounded-lg"
                flowID={useSearchParams().get("flowID") ?? undefined}
              />
          </div>
      </div>
  );
}
