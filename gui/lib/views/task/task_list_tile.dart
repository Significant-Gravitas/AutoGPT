import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';

class TaskListTile extends StatefulWidget {
  final Task task;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const TaskListTile({
    Key? key,
    required this.task,
    required this.onTap,
    required this.onDelete,
  }) : super(key: key);

  @override
  _TaskListTileState createState() => _TaskListTileState();
}

class _TaskListTileState extends State<TaskListTile> {
  bool _isSelected = false;

  @override
  Widget build(BuildContext context) {
    // Determine the width of the TaskView
    double taskViewWidth = MediaQuery.of(context).size.width;
    double tileWidth = taskViewWidth - 20;
    if (tileWidth > 260) {
      tileWidth = 260;
    }

    return GestureDetector(
      onTap: () {
        setState(() {
          _isSelected = !_isSelected;
        });
        widget.onTap();
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
              color: _isSelected ? Colors.grey[300] : Colors.white,
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
                    widget.task.title,
                    style: const TextStyle(color: Colors.black),
                  ),
                ),
                // If the task is selected, show a delete icon
                if (_isSelected)
                  IconButton(
                    splashRadius: 0.1,
                    icon: const Icon(Icons.close, color: Colors.black),
                    onPressed: widget.onDelete,
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
