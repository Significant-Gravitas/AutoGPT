import { router } from "expo-router";
import { Linking } from "react-native";

import { homeLinkTarget } from "./helpers";

export function openHomeLink(href: string) {
  const target = homeLinkTarget(href);
  if (!target) return;
  if (target.kind === "chat") {
    router.push(`/chat/${target.sessionID}`);
  } else if (target.kind === "team") {
    router.navigate("/team");
  } else {
    Linking.openURL(target.url).catch(() => undefined);
  }
}
