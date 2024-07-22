"use client";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import FlowEditor from '@/components/Flow';

export default function Home() {
  const query = useSearchParams();

  return (
      <div className="flex flex-col items-center min-h-screen">
          <div className="z-10 w-full flex items-center justify-between font-mono text-sm relative">
              <p className="border border-gray-600 rounded-xl pb-4 pt-4 p-4">
                  Get started by adding a&nbsp;
                  <code className="font-mono font-bold">block</code>
              </p>
              <div className="absolute top-0 right-0 p-4">
                  <a
                      className="pointer-events-auto flex place-items-center gap-2"
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
                flowID={query.get("flowID") ?? query.get("templateID") ?? undefined}
                template={!!query.get("templateID")}
              />
          </div>
      </div>
  );
}
