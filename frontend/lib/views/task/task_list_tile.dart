import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';

class TaskListTile extends StatelessWidget {
  final Task task;
  final VoidCallback onTap;
  final VoidCallback onDelete;
  final bool selected;

  const TaskListTile({
    Key? key,
    required this.task,
    required this.onTap,
    required this.onDelete,
    this.selected = false,
  }) : super(key: key);

  Widget build(BuildContext context) {
    // Determine the width of the TaskView
    double taskViewWidth = MediaQuery.of(context).size.width;
    double tileWidth = taskViewWidth - 20;
    if (tileWidth > 260) {
      tileWidth = 260;
    }

    return GestureDetector(
      onTap: () {
        onTap();
      },
      child: Material(
        // Use a transparent color to avoid any unnecessary color overlay
        color: Colors.transparent,
        child: Padding(
          // Provide a horizontal padding to ensure the tile does not touch the edges
          padding: const EdgeInsets.symmetric(horizontal: 10.0),
          child: Container(
            // Width and height specifications for the tile
            width: tileWidth,
            height: 50,
            decoration: BoxDecoration(
              // Use conditional operator to determine background color based on selection
              color: selected ? Colors.grey[300] : Colors.white,
              borderRadius: BorderRadius.circular(8.0),
            ),
            child: Row(
              children: [
                // Space from the left edge of the tile
                const SizedBox(width: 8),
                // Message bubble icon indicating a task
                const Icon(Icons.messenger_outline, color: Colors.black),
                const SizedBox(width: 8),
                // Task title
                Expanded(
                  child: Text(
                    task.title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: Colors.black),
                  ),
                ),
                // If the task is selected, show a delete icon
                if (selected)
                  IconButton(
                    splashRadius: 0.1,
                    icon: const Icon(Icons.close, color: Colors.black),
                    onPressed: onDelete,
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
