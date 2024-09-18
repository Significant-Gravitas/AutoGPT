# AutoGPT Flutter Client

## Description

This repository contains the Flutter client for the AutoGPT project. The application facilitates users in discussing various tasks with a single agent. The app is built to be cross-platform and runs on Web, Android, iOS, Windows, and Mac.

## Features

- List and manage multiple tasks.
- Engage in chat conversations related to selected tasks.

## Design document

The design document for this project provides a detailed outline of the architecture, components, and other important aspects of this application. Please note that this is a living, growing document and it is subject to change as the project evolves.

You can access the design document [here](https://docs.google.com/document/d/1S-o2np1gq5JwFq40wPHDUVLi-mylz4WMvCB8psOUjc8/).

## Requirements

- Flutter 3.x
- Dart 3.x

Flutter comes with Dart, to install Flutter, follow the instructions here: https://docs.flutter.dev/get-started/install

## Installation

1. **Clone the repo:**
```
git clone https://github.com/Significant-Gravitas/AutoGPT.git
```

2. **Navigate to the project directory:**
```
cd AutoGPT/frontend
```

3. **Get Flutter packages:**
```
flutter pub get
```

4. **Run the app:**
```
#For chromium users on linux:
#export CHROME_EXECUTABLE=/usr/bin/chromium
flutter run -d chrome --web-port 5000
```

## Project Structure

- `lib/`: Contains the main source code for the application.
- `models/`: Data models that define the structure of the objects used in the app.
- `views/`: The UI components of the application.
- `viewmodels/`: The business logic and data handling for the views.
- `services/`: Contains the service classes that handle communication with backend APIs and other external data sources. These services are used to fetch and update data that the app uses, and they are consumed by the ViewModels.
- `test/`: Contains the test files for unit and widget tests.

## Responsive Design

The app features a responsive design that adapts to different screen sizes and orientations. On larger screens (Web, Windows, Mac), views are displayed side by side horizontally. On smaller screens (Android, iOS), views are displayed in a tab bar controller layout.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
