import 'package:flutter/material.dart';

class TaskView extends StatelessWidget {
  const TaskView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.blue,
      child: const Center(
        child: Text(
          'Tasks',
          style: TextStyle(fontSize: 24, color: Colors.white),
        ),
      ),
    );
  }
}
