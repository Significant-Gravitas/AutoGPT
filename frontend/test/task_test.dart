import 'package:auto_gpt_flutter_client/models/task.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Task', () {
    // Test the properties of the Task class
    test('Task properties', () {
      final task = Task(id: 1, title: 'Test Task');

      expect(task.id, 1);
      expect(task.title, 'Test Task');
    });

    // Test Task.fromMap method
    test('Task.fromMap', () {
      final task = Task.fromMap({'id': 1, 'title': 'Test Task'});

      expect(task.id, 1);
      expect(task.title, 'Test Task');
    });

    // Test creating a Task with an empty title
    test('Task with empty title', () {
      expect(() => Task(id: 2, title: ''), throwsA(isA<AssertionError>()));
    });

    // Test that two Task objects with the same id and title are equal
    test('Two tasks with same properties are equal', () {
      final task1 = Task(id: 4, title: 'Same Task');
      final task2 = Task(id: 4, title: 'Same Task');

      expect(task1, task2);
    });

    // Test that toString() returns a string representation of the Task
    test('toString returns string representation', () {
      final task = Task(id: 5, title: 'Test toString');

      expect(task.toString(), 'Task(id: 5, title: Test toString)');
    });

    // Test that title of Task can be modified
    test('Modify task title', () {
      final task = Task(id: 6, title: 'Initial Title');
      task.title = 'Modified Title';

      expect(task.title, 'Modified Title');
    });

    // Test that setting an empty title throws an error
    test('Set empty title', () {
      final task = Task(id: 7, title: 'Valid Title');

      expect(() => task.title = '', throwsA(isA<ArgumentError>()));
    });
  });
}
