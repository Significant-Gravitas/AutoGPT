# AutoGPT mobile apps

This draft introduces iOS and Android apps for AutoGPT's existing hosted chat interfaces. The web application remains responsible for conversations, streaming responses, agents, tools, attachments, authentication, and account settings. Improvements to that application reach the mobile apps without a store release.

The mobile projects will provide only the operating-system integration needed to use that experience: a persistent web session, safe navigation, keyboard and safe-area handling, attachment pickers, loading and recovery states, and platform navigation. They will not introduce a separate chat API client or fork the web chat components.

## Scope

- Target an iPhone 17 Pro and a comparable current Android flagship.
- Use the installed iOS simulator first; avoid downloading Android emulator images on this storage-constrained development machine.
- Keep native dependencies small and document reproducible build commands.
- Verify first-party sign-in and identify social authentication limitations explicitly.
- Keep the broader responsive-web redesign in its separate workstream.
- Keep the pull request a draft. Store publication, signing credentials, production deployment, and public release are outside this prototype.

## Validation plan

1. Build and launch the iOS app in the available simulator and capture progress screenshots.
2. Build the Android app when the available SDK permits; separately record device checks that were not run.
3. Test navigation policy against deceptive hosts, unsafe URL schemes, and external links.
4. Exercise loading, offline recovery, back navigation, session persistence, keyboard behavior, and attachment selection.
5. Verify the app loads the hosted chat and that web updates do not require copying chat implementation into either mobile project.

Implementation and verified results will be added to this document as the draft develops.
