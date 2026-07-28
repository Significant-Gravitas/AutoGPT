"use client";

import type { ToolUIPart } from "ai";
import {
  KeyIcon,
  PlugsConnectedIcon,
  UserIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { ToolErrorCard } from "../../components/ToolErrorCard/ToolErrorCard";
import {
  ContentCard,
  ContentCardHeader,
  ContentCardTitle,
  ContentGrid,
  ContentHint,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import {
  formatProviderName,
  getAnimationText,
  getCredentialTypeLabel,
  getListCredentialsOutput,
  isCredentialList,
  isErrorOutput,
  type CredentialMeta,
} from "./helpers";

interface Props {
  part: ToolUIPart;
}

function ToolStatusIcon({
  isStreaming,
  isError,
}: {
  isStreaming: boolean;
  isError: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }
  if (isStreaming) {
    return <OrbitLoader size={14} />;
  }
  return (
    <PlugsConnectedIcon
      size={14}
      weight="regular"
      className="text-neutral-400"
    />
  );
}

function CredentialCard({ credential }: { credential: CredentialMeta }) {
  return (
    <ContentCard>
      <ContentCardHeader>
        <div className="flex items-center gap-2">
          <KeyIcon size={14} weight="fill" className="text-neutral-600" />
          <ContentCardTitle>
            {formatProviderName(credential.provider)}
          </ContentCardTitle>
        </div>
      </ContentCardHeader>
      <ContentHint>
        {getCredentialTypeLabel(credential.type)}
        {credential.title ? ` · ${credential.title}` : ""}
        {credential.host ? ` · ${credential.host}` : ""}
        {credential.is_managed ? " · Managed by AutoGPT" : ""}
      </ContentHint>
      {credential.username && (
        <div className="mt-1 flex items-center gap-1.5">
          <UserIcon size={12} weight="duotone" className="text-neutral-600" />
          <span className="text-xs text-zinc-600">{credential.username}</span>
        </div>
      )}
      {credential.scopes && credential.scopes.length > 0 && (
        <div className="mt-1">
          <span className="text-xs text-zinc-500">
            Scopes: {credential.scopes.join(", ")}
          </span>
        </div>
      )}
    </ContentCard>
  );
}

export function ListCredentialsTool({ part }: Props) {
  const text = getAnimationText(part);
  const output = getListCredentialsOutput(part);

  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const list =
    part.state === "output-available" && output && isCredentialList(output)
      ? output
      : null;

  const errorOutput = output && isErrorOutput(output) ? output : null;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolStatusIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {isError && (
        <div className="mt-2">
          <ToolErrorCard
            message={errorOutput?.message}
            fallbackMessage="Could not check connected integrations."
            error={errorOutput?.error}
            actions={[]}
          />
        </div>
      )}

      {list && (
        <ToolAccordion
          icon={<PlugsConnectedIcon size={32} weight="light" />}
          title={
            list.count > 0
              ? `${list.count} connected integration${list.count !== 1 ? "s" : ""}`
              : "No connected integrations"
          }
          defaultExpanded
        >
          {!list.provisioning_complete && (
            <ContentHint>
              This list may be incomplete — some managed integrations could not
              be checked.
            </ContentHint>
          )}
          {list.credentials.length > 0 ? (
            <ContentGrid className="sm:grid-cols-2">
              {list.credentials.map((credential) => (
                <CredentialCard key={credential.id} credential={credential} />
              ))}
            </ContentGrid>
          ) : (
            <ContentMessage>{list.message}</ContentMessage>
          )}
        </ToolAccordion>
      )}
    </div>
  );
}
