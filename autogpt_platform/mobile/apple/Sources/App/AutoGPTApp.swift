import SwiftUI

@main
struct AutoGPTApp: App {
  @State private var session = SessionStore()

  var body: some Scene {
    WindowGroup {
      #if UI_PROTOCOL_FIXTURES
        if FixtureLaunch.isRequested {
          FixtureCatalogView()
        } else {
          AppRootView(session: session)
        }
      #else
        AppRootView(session: session)
      #endif
    }
  }
}

/// Chooses between restoring, the sign-in flow, and the signed-in app. The
/// signed-in app is rebuilt for every session generation, so nothing from a
/// previous account or server survives a switch.
struct AppRootView: View {
  let session: SessionStore

  var body: some View {
    Group {
      switch session.phase {
      case .restoring:
        Color(uiColor: .systemBackground).ignoresSafeArea()
      case .signedOut:
        AuthFlowView(session: session)
      case .signedIn(let context):
        MainTabView(context: context, session: session)
          .id(context.generation)
      }
    }
    .task { await session.restore() }
  }
}
