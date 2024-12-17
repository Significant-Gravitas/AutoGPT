# 🚀 Contributing to the Docs

Welcome to our documentation contribution guide! We're excited to have you here. This guide will help you get started with contributing to our documentation. Let's make our docs better together! 💪

<div align="center">

![Documentation Contributors](https://img.shields.io/github/contributors/Significant-Gravitas/AutoGPT?style=for-the-badge)
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)
![Documentation Build Status](https://img.shields.io/badge/docs-passing-success?style=for-the-badge)

</div>

## 🛠️ Setting up the Docs

### Prerequisites

- Git installed on your machine
- Python 3.8 or higher
- A text editor of your choice

### Step-by-Step Setup Guide

1. **Clone the Repository** 📥
   ```shell
   git clone github.com/Significant-Gravitas/AutoGPT.git
   ```

2. **Install Dependencies** 📚
   ```shell
   python -m pip install -r docs/requirements.txt
   ```
   or
   ```shell
   python3 -m pip install -r docs/requirements.txt
   ```

3. **Start the Development Server** 🔥
   ```shell
   mkdocs serve
   ```

4. **View the Docs** 🌐
   - Open your browser
   - Navigate to `http://127.0.0.1:8000`
   - Changes will auto-reload! 🔄

## 📝 Adding New Content

### Creating a New Page

1. **Create the File** 📄
   - Navigate to `docs/content`
   - Create a new markdown file

2. **Update Navigation** 🗺️
   - Open `mkdocs.yml`
   - Add your page to the `nav` section

3. **Add Content** ✍️
   - Write your content using Markdown
   - Add images and code examples as needed

4. **Preview Changes** 👀
   - Run `mkdocs serve`
   - Check your new page in the browser

## 🔍 Quality Checks

### Link Validation
```shell
mkdocs build
```
Watch for any warnings about broken links in the console output! 🚨

## 🎯 Submitting Your Contribution

### Pull Request Process

1. **Create a Branch** 🌿
   - Use a descriptive name for your branch
   - Keep changes focused and atomic

2. **Submit PR** 📮
   - Fill out the PR template
   - Add screenshots if applicable
   - Link related issues

3. **Review Process** 👥
   - Address reviewer feedback
   - Make requested changes
   - Maintain active communication

## ⭐ Best Practices

- Keep documentation clear and concise
- Use proper Markdown formatting
- Include code examples where helpful
- Test all links before submitting
- Follow existing documentation style

---

<div align="center">

**Thank you for contributing to our documentation!** ❤️

*Together, we make documentation better for everyone.*

</div>
