import 'package:auto_gpt_flutter_client/views/agent_view.dart';
import 'package:auto_gpt_flutter_client/views/chat_view.dart';
import 'package:flutter/cupertino.dart';

class MainLayout extends StatelessWidget {
  const MainLayout({super.key});

  @override
  Widget build(BuildContext context) {
    // Get the screen width
    double width = MediaQuery.of(context).size.width;

    // Check the screen width and return the appropriate layout
    if (width > 800) {
      // For larger screens, return a side-by-side layout
      return const Row(
        children: [
          SizedBox(width: 280, child: AgentView()),
          Expanded(child: ChatView()),
        ],
      );
    } else {
      // For smaller screens, return a tabbed layout
      return CupertinoTabScaffold(
        tabBar: CupertinoTabBar(
          items: const <BottomNavigationBarItem>[
            BottomNavigationBarItem(
              icon: Icon(CupertinoIcons.person),
              label: 'Agents',
            ),
            BottomNavigationBarItem(
              icon: Icon(CupertinoIcons.chat_bubble),
              label: 'Chat',
            ),
          ],
        ),
        tabBuilder: (BuildContext context, int index) {
          CupertinoTabView? returnValue;

          switch (index) {
            case 0:
              returnValue = CupertinoTabView(builder: (context) {
                return const CupertinoPageScaffold(
                  child: AgentView(),
                );
              });
              break;
            case 1:
              returnValue = CupertinoTabView(builder: (context) {
                return const CupertinoPageScaffold(
                  child: ChatView(),
                );
              });
              break;
          }

          return returnValue ??
              CupertinoTabView(builder: (context) {
                return CupertinoPageScaffold(
                  child: Container(), // Default empty container
                );
              });
        },
      );
    }
  }
}
