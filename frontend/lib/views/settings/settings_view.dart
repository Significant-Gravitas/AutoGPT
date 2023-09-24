import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

/// [SettingsView] displays a list of settings that the user can configure.
/// It uses [SettingsViewModel] for state management and logic.
class SettingsView extends StatelessWidget {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context) {
    // [ChangeNotifierProvider] provides an instance of [SettingsViewModel] to the widget tree.
    return ChangeNotifierProvider(
      create: (context) => SettingsViewModel(),
      child: Scaffold(
        appBar: AppBar(
            backgroundColor: Colors.grey,
            foregroundColor: Colors.black,
            title: const Text('Settings')),
        body: Consumer<SettingsViewModel>(
          builder: (context, viewModel, child) {
            // A list of settings is displayed using [ListView].
            return ListView(
              children: [
                // Dark Mode Toggle
                SwitchListTile(
                  title: const Text('Dark Mode'),
                  value: viewModel.isDarkModeEnabled,
                  onChanged: viewModel.toggleDarkMode,
                ),
                // Developer Mode Toggle
                SwitchListTile(
                  title: const Text('Developer Mode'),
                  value: viewModel.isDeveloperModeEnabled,
                  onChanged: viewModel.toggleDeveloperMode,
                ),
                // Base URL Configuration
                ListTile(
                  title: const Text('Base URL'),
                  subtitle: TextFormField(
                    initialValue: viewModel.baseURL,
                    onChanged: viewModel.updateBaseURL,
                    decoration: const InputDecoration(
                      hintText: 'Enter Base URL',
                    ),
                  ),
                ),
                // Continuous Mode Steps Configuration
                ListTile(
                  title: const Text('Continuous Mode Steps'),
                  // User can increment or decrement the number of steps using '+' and '-' buttons.
                  subtitle: Row(
                    children: [
                      IconButton(
                        icon: const Icon(Icons.remove),
                        onPressed: viewModel
                            .decrementContinuousModeSteps, // Decrement the number of steps.
                      ),
                      Text('${viewModel.continuousModeSteps} Steps'),
                      IconButton(
                        icon: const Icon(Icons.add),
                        onPressed: viewModel
                            .incrementContinuousModeSteps, // Increment the number of steps.
                      ),
                    ],
                  ),
                ),
                ListTile(
                  title: Text('Sign Out'),
                  onTap: () {
                    viewModel.signOut();
                    // Optionally, navigate to a different view or show a message
                  },
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}
