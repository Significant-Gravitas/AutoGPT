import 'package:auto_gpt_flutter_client/views/task_view.dart';
import 'package:auto_gpt_flutter_client/views/chat_view.dart';
import 'package:flutter/cupertino.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart'; // Importing the TaskViewModel
import 'package:provider/provider.dart'; // Required for accessing the ViewModel

class MainLayout extends StatelessWidget {
  const MainLayout({Key? key}) : super(key: key); // Corrected the constructor

  @override
  Widget build(BuildContext context) {
    // Get the screen width
    double width = MediaQuery.of(context).size.width;

    // Access the TaskViewModel from the context
    final taskViewModel = Provider.of<TaskViewModel>(context);

    // Check the screen width and return the appropriate layout
    if (width > 800) {
      // For larger screens, return a side-by-side layout
      return Row(
        children: [
          SizedBox(
              width: 280,
              child: TaskView(
                  viewModel:
                      taskViewModel)), // Pass the viewModel to the TaskView
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
              label: 'Tasks',
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
                return CupertinoPageScaffold(
                  child: TaskView(
                      viewModel:
                          taskViewModel), // Pass the viewModel to the TaskView
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
