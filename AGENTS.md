# AGENTS.md

## Git

- 変更を加えたらその都度コミットする。コミットを依頼されるまで溜めない。
- `main` に直接コミットする。トピックブランチは作らない。
- push は依頼されたときだけ行う。
- 1 タスク = 1 コミット。複数の作業をまとめた大きなコミットにしない。
- バグ修正はリファクタリングとは別コミットにし、先にコミットする。

### コミットメッセージ

[Conventional Commits](https://www.conventionalcommits.org/) 形式で英語で書く。

- type: `feat` / `fix` / `refactor` / `style` / `docs` / `build` など
- scope はモジュール名（例: `client`, `server`, `train`, `init`, `logger`, `app`, `setup_data`）。複数モジュールにまたがる場合は省略する。
- コマンドやパスが変わる破壊的変更は `!` を付け、`BREAKING CHANGE:` フッターを書く。

例:

```
feat(client): add proximal term to local training
fix(server): aggregate each center from its assigned clients only
refactor!: move code into the src/mcfl package
```
