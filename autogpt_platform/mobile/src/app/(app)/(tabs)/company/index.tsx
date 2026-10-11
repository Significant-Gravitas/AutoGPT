import { router, Stack, useIsFocused } from "expo-router";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
} from "react-native";

import type { ExpertProfile } from "../../../../api/experts";
import { describeChatError } from "../../../../api/chat";
import { Avatar } from "../../../../design/components/Avatar";
import { Row, Section } from "../../../../design/components/Grouped";
import { InlineNotice } from "../../../../design/components/StateViews";
import { colors, spacing, type } from "../../../../design/theme";
import { formatListTimestamp } from "../../../../lib/time";
import { useExperts } from "../../../../features/app/AppDataProvider";
import { useSignedIn } from "../../../../features/auth/SessionProvider";
import {
  expertConversationHref,
  expertParticipant,
  newConversationHref,
  ottoParticipant,
} from "../../../../features/company/participants";

export default function CompanyScreen() {
  const { api, server } = useSignedIn();
  const { state, directory } = useExperts();
  const [opening, setOpening] = useState<string | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  const otto = ottoParticipant(server.authUrl);

  const focused = useIsFocused();
  useEffect(() => {
    if (focused) void directory.load();
  }, [focused, directory]);

  async function openExpert(expert: ExpertProfile) {
    if (opening) return;
    setOpening(expert.id);
    setOpenError(null);
    try {
      router.push(await expertConversationHref(api, expert.id));
    } catch (error) {
      setOpenError(describeChatError(error));
    } finally {
      setOpening(null);
    }
  }

  return (
    <>
      <Stack.Toolbar placement="left">
        <Stack.Toolbar.Button
          icon="person.crop.circle"
          accessibilityLabel="Account"
          onPress={() => router.push("/account")}
        />
      </Stack.Toolbar>
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={styles.screen}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={state.status === "loading"}
            onRefresh={() => void directory.load()}
          />
        }
      >
        {openError ? <InlineNotice text={openError} tone="danger" /> : null}
        <Section
          header="Always here"
          footer="Otto runs your AI company and can start anything for you."
        >
          <Row
            title={otto.name}
            subtitle={otto.role}
            leading={<Avatar name={otto.name} uri={otto.avatarURL} size={44} />}
            showsChevron
            accessibilityHint="Starts a new conversation with Otto"
            onPress={() => router.push(newConversationHref(null))}
            testID="company-otto"
          />
        </Section>
        <Section
          header="Your experts"
          footer={
            state.roster.length > 0
              ? "Opens your most recent conversation with that expert."
              : undefined
          }
        >
          {state.roster.length > 0 ? (
            state.roster.map((expert) => {
              const participant = expertParticipant(
                expert,
                expert.id,
                state.status,
                server.authUrl,
              );
              return (
                <Row
                  key={expert.id}
                  title={expert.name}
                  subtitle={[participant.role, expertStatus(expert)]
                    .filter(Boolean)
                    .join(" · ")}
                  leading={
                    <Avatar
                      name={expert.name}
                      uri={participant.avatarURL}
                      size={44}
                    />
                  }
                  trailing={
                    opening === expert.id ? <ActivityIndicator /> : null
                  }
                  showsChevron
                  accessibilityHint={`Opens your conversation with ${expert.name}`}
                  onPress={() => void openExpert(expert)}
                  testID={`company-expert-${expert.id}`}
                />
              );
            })
          ) : (
            <Row title={emptyRosterText(state.status, state.error)} />
          )}
        </Section>
        {state.error && state.roster.length > 0 ? (
          <InlineNotice text={state.error} tone="danger" />
        ) : null}
        <Text style={[type.footnote, styles.note]}>
          Hiring and changing experts happens in conversation with Otto.
          Confirming those changes isn&apos;t available in the app yet.
        </Text>
      </ScrollView>
    </>
  );
}

function expertStatus(expert: ExpertProfile): string | null {
  if (expert.setupStatus && expert.setupStatus !== "ready") return "Setting up";
  if (expert.schedulesPausedAt) return "Schedules paused";
  if (expert.lastRunAt) {
    const when = formatListTimestamp(expert.lastRunAt);
    return expert.lastRunStatus
      ? `Last run ${when}, ${expert.lastRunStatus.toLowerCase()}`
      : `Last run ${when}`;
  }
  return null;
}

function emptyRosterText(status: string, error: string | null): string {
  if (status === "loading" || status === "idle") return "Loading your experts…";
  if (status === "error") return error ?? "Couldn't load your experts.";
  return "No experts hired yet.";
}

const styles = StyleSheet.create({
  screen: { backgroundColor: colors.groupedBackground },
  content: { paddingBottom: spacing.xxl },
  note: {
    color: colors.secondaryLabel,
    marginHorizontal: spacing.xl,
    marginTop: spacing.xl,
  },
});
