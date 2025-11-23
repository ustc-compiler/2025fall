# USTC _Principles and Techniques of Compiler(H)_ 2025 course homepage

Homepage link: <https://ustc-compiler.github.io/2025fall>

This README is a toturial of how to write and preview the docs.

## Development

First install dependencies and activate environment:

```bash
# It's recommended to install in a virtual environment.
uv sync
source .venv/bin/activate # .venv/Scripts/activate.ps1 in Windows
```

Then make your modifications. You can preview your changes by running:

```bash
mkdocs serve
```

### format

In order to keep the doc style consistent, we use [Prettier](https://prettier.io/) and [AutoCorrect](https://github.com/huacnlee/autocorrect).

```bash
# you need Node.js installed first
npm install # then install Prettier and AutoCorrect
```

See [Prettier Doc: Editor Integration](https://prettier.io/docs/en/editors.html) and [AutoCorrect: VS Code Extension](https://github.com/huacnlee/autocorrect#vs-code-extension) for editor intergration.

> Type Ctrl+Shift+P and "show recommended extensions" to see these extensions and install them. VSCode might pop up a window. Just install these extensions.

You should **format markdown files before commit**:

```bash
# see .prettierignore for ignoring certain files or folders
npm run test    # to see if there is a format error
npm run format  # to format all files under docs/
```

### File Structure

`mkdocs.yml` is the configuration file. When you want to add a new page, you need to add it to `nav` in `mkdocs.yml`.

`docs/` is the directory of all Markdown files. You can create subdirectories to organize your files.

### Admonitions

Read <https://squidfunk.github.io/mkdocs-material/reference/admonitions/> for more details.

## Deployment

1. Enable GitHub Actions.
2. Push an commit to trigger the workflow, create the branch gh-pages and generete pages.
3. In Settings > Pages > Build and deployment
   1. select the "Deploy from a branch" option in "Source"
   2. select "gh-pages" in "Branch"

## License

All texts are All Rights Reserved by default.
