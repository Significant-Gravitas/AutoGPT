import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/mock_data.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('TaskViewModel', () {
    late TaskViewModel viewModel;

    setUp(() {
      viewModel = TaskViewModel();
    });

    test('Fetches tasks successfully', () {
      viewModel.fetchTasks();
      expect(viewModel.tasks, isNotEmpty);
    });

    test('Selects a task successfully', () {
      viewModel.fetchTasks();
      viewModel.selectTask(1);
      expect(viewModel.selectedTask, isNotNull);
    });

    test(
        'Notifiers are properly telling UI to update after fetching a task or selecting a task',
        () {
      bool hasNotified = false;
      viewModel.addListener(() {
        hasNotified = true;
      });

      viewModel.fetchTasks();
      expect(hasNotified, true);

      hasNotified = false; // Reset for next test
      viewModel.selectTask(1);
      expect(hasNotified, true);
    });

    test('No tasks are fetched', () {
      // Clear mock data for this test
      mockTasks.clear();

      viewModel.fetchTasks();
      expect(viewModel.tasks, isEmpty);
    });

    test('No task is selected', () {
      expect(viewModel.selectedTask, isNull);
    });

    test('Creates a task successfully', () {
      final initialCount = viewModel.tasks.length;
      viewModel.createTask('New Task');
      expect(viewModel.tasks.length, initialCount + 1);
    });

    test('Deletes a task successfully', () {
      viewModel.fetchTasks();
      final initialCount = viewModel.tasks.length;
      viewModel.deleteTask(1);
      expect(viewModel.tasks.length, initialCount - 1);
    });

    test('Deletes a task with invalid id', () {
      // TODO: Update this test to expect an error once we have TaskService implemented
      final initialCount = viewModel.tasks.length;
      viewModel.deleteTask(9999); // Assuming no task with this id exists
      expect(viewModel.tasks.length, initialCount); // Count remains same
    });

    test('Select a task that doesn\'t exist', () {
      expect(() => viewModel.selectTask(9999), throwsA(isA<ArgumentError>()));
    });
  });
}
