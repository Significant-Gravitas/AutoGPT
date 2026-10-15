import { router } from "expo-router";
import { Alert, ScrollView, StyleSheet, Text, View } from "react-native";

import { Row, Section } from "../../design/components/Grouped";
import { Icon } from "../../design/components/Icon";
import { InlineNotice } from "../../design/components/StateViews";
import { colors, spacing, type } from "../../design/theme";
import { useSession } from "../../features/auth/SessionProvider";
import {
  CLOUD_SERVER,
  hostLabel,
  type ServerConfig,
} from "../../features/servers/serverConfig";

export default function WelcomeScreen() {
  const { state, customServers, chooseServer, forgetServer } = useSession();
  const notice = state.status === "signedOut" ? state.notice : null;

  async function continueWith(server: ServerConfig) {
    await chooseServer(server);
    router.push("/sign-in");
  }

  function confirmForget(server: ServerConfig) {
    Alert.alert(`Forget ${server.name}?`, "You can add it again later.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Forget",
        style: "destructive",
        onPress: () => void forgetServer(server.id),
      },
    ]);
  }

  return (
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={styles.content}
      style={styles.screen}
    >
      <View style={styles.hero}>
        <Icon name="sparkles" size={48} color={colors.tint} />
        <Text
          style={[type.largeTitle, styles.title]}
          accessibilityRole="header"
        >
          AutoGPT
        </Text>
        <Text style={[type.body, styles.subtitle]}>
          Talk with Otto and the experts in your AI company.
        </Text>
      </View>
      {notice ? <InlineNotice text={notice} /> : null}
      <Section header="Where's your account?">
        <Row
          title={CLOUD_SERVER.name}
          subtitle={hostLabel(CLOUD_SERVER.authUrl)}
          leading={<Icon name="cloud" size={22} color={colors.tint} />}
          showsChevron
          onPress={() => void continueWith(CLOUD_SERVER)}
          testID="choose-cloud"
        />
        <Row
          title="Your own server"
          subtitle="A self-hosted AutoGPT"
          leading={<Icon name="server.rack" size={22} color={colors.tint} />}
          showsChevron
          onPress={() => router.push("/own-server")}
          testID="choose-own-server"
        />
      </Section>
      {customServers.length > 0 ? (
        <Section
          header="Recent servers"
          footer="Touch and hold a server to forget it."
        >
          {customServers.map((server) => (
            <Row
              key={server.id}
              title={server.name}
              subtitle={hostLabel(server.authUrl)}
              leading={
                <Icon name="network" size={22} color={colors.secondaryLabel} />
              }
              showsChevron
              onPress={() => void continueWith(server)}
              onLongPress={() => confirmForget(server)}
              longPressLabel="Forget server"
            />
          ))}
        </Section>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { backgroundColor: colors.groupedBackground },
  content: { paddingBottom: spacing.xxl },
  hero: {
    alignItems: "center",
    gap: spacing.sm,
    paddingTop: spacing.xxl * 2,
    paddingHorizontal: spacing.xl,
  },
  title: { color: colors.label },
  subtitle: { color: colors.secondaryLabel, textAlign: "center" },
});
