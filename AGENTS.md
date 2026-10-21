# AGENTS.md

## Git

- 変更を加えたらその都度コミットする。コミットを依頼されるまで溜めない。
- `main` に直接コミットする。トピックブランチは作らない。
- push は依頼されたときだけ行う。
- 1 タスク = 1 コミット。複数の作業をまとめた大きなコミットにしない。
- バグ修正はリファクタリングとは別コミットにし、先にコミットする。

### コミットメッセージ

Conventional Commits 形式で英語で書く。scope はモジュール名（`client`, `server`, `train` など）にし、複数モジュールにまたがる場合は省略する。

例: `feat(client): add proximal term to local training`
