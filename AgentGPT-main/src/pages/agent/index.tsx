import { type NextPage } from "next";
import DefaultLayout from "../../layout/default";
import Button from "../../components/Button";

import React, { useState } from "react";
import { useRouter } from "next/router";
import { api } from "../../utils/api";
import ChatWindow from "../../components/ChatWindow";
import type { Message } from "../../types/agentTypes";
import Toast from "../../components/toast";
import { FaTrash, FaShare, FaBackspace } from "react-icons/fa";
import { env } from "../../env/client.mjs";

import { useTranslation } from 'react-i18next';

const AgentPage: NextPage = () => {
  const [ t ] = useTranslation();
  const [showCopied, setShowCopied] = useState(false);
  const router = useRouter();

  const agentId = typeof router.query.id === "string" ? router.query.id : "";

  const getAgent = api.agent.findById.useQuery(agentId, {
    enabled: router.isReady,
  });

  const deleteAgent = api.agent.deleteById.useMutation({
    onSuccess: () => {
      void router.push("/");
    },
  });

  const messages = getAgent.data ? (getAgent.data.tasks as Message[]) : [];

  const shareLink = () => {
    return encodeURI(`${env.NEXT_PUBLIC_VERCEL_URL as string}${router.asPath}`);
  };

  return (
    <DefaultLayout
      className="flex w-full flex-col items-center justify-center gap-4 p-2 sm:p-4"
      centered
    >
      <ChatWindow
        messages={messages}
        title={getAgent?.data?.name}
        showDonation={false}
        className="min-h-[80vh] md:w-[80%]"
        fullscreen
      />
      <div className="flex flex-row gap-2">
        <Button
          icon={<FaShare />}
          onClick={() => {
            void window.navigator.clipboard
              .writeText(shareLink())
              .then(() => setShowCopied(true));
          }}
          enabledClassName={"bg-green-600 hover:bg-green-400"}
        >
          Share
        </Button>
        <Button
          icon={<FaTrash />}
          onClick={() => {
            deleteAgent.mutate(agentId);
          }}
          enabledClassName={"bg-red-600 hover:bg-red-400"}
        >
          Delete
        </Button>
        <Button icon={<FaBackspace />} onClick={() => void router.push("/")}>
          Back
        </Button>
      </div>
      <Toast
        model={[showCopied, setShowCopied]}
        title={t('Copied to clipboard! 🚀')}
        className="bg-gray-950 text-sm"
      />
    </DefaultLayout>
  );
};

export default AgentPage;
