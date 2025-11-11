## AutoPMoB Agent（日本語版）

このリポジトリは、PDF から数理モデル構築に必要な変数・方程式を抽出し、目的に沿った候補モデルを
LangChain + OpenAI で自動生成するワークフローを提供します。中心となるスクリプトは
`scripts/model_builder_agent.py` です。

### パイプライン概要

1. **Extractor** – PDF を読み込み、ナラティブ・変数・方程式を抽出。
2. **DBOrganizer** – 文書横断で変数・方程式を正規化し、同義語をまとめる。
3. **ModelBuilder** – 整理済みデータから目的に合致する候補モデルを複数生成。

各工程のプロンプトは `prompts/` 以下の YAML で管理します。

---

## ディレクトリ構成

```
autopmob-agent/
├── configs/          # ワークフロー設定ファイル（YAML）
├── data/             # PDF などの入力データ
├── prompts/          # 各工程のプロンプト (system/human を含む YAML)
├── results/          # 実行結果（自動生成）
├── scripts/
│   └── model_builder_agent.py
├── README.md
└── README_JP.md
```

---

## プロンプトファイル

各工程ごとに 1 つの YAML ファイルを用意し、`system` と `human` を記述します。

```yaml
system: |-
  システムロールの指示...
human: |-
  人間ロールのテンプレート（{objective} などのプレースホルダ可）
```

例:

```
prompts/
  extractor.yaml
  organizer.yaml
  model_builder.yaml
```

工程ごとの必須プレースホルダ:

- **Extractor**: `{objective}`, `{doc_id}`, `{title}`
- **DBOrganizer**: `{objective}`, `{documents}`
- **ModelBuilder**: `{objective}`, `{db_snapshot}`, `{num_models}`

---

## 設定ファイル（configs/*.yaml）

最低限必要な項目:

```yaml
pdf:
  - path: data/.../paper1.pdf
    doc_id: cstr-1

objective:
  description: >
    モデル化の目的
  input_variables: [...]
  output_variables: [...]
  success_criteria: >
    期待するモデルの性質

max_models: 3
model_name: gpt-5
temperature: 0.2

prompts:
  extractor: prompts/extractor.yaml
  organizer: prompts/organizer.yaml
  model_builder: prompts/model_builder.yaml
```

- `pdf.path` はリポジトリルートからの相対パス、または絶対パスを指定できます。
- `prompts` は必須で、各工程の YAML ファイルを指します（スクリプト内にデフォルトはありません）。

---

## 実行方法

```bash
uv run python scripts/model_builder_agent.py \
  --config configs/cstr-sample.yaml \
  --run-name cstr_run_001 \
  --save-stage-io
```

### 主な引数

| フラグ | 説明 |
|--------|------|
| `--config`      | **必須。** 実行設定 YAML のパス。 |
| `--results-dir` | 結果フォルダの親ディレクトリ（既定 `results/`）。 |
| `--run-name`    | 実行フォルダ名（未指定ならタイムスタンプ）。 |
| `--save-stage-io` | 指定すると各工程の入出力 JSON を `artifacts/` に保存。 |

---

## 出力

`results/<run_name>/` ディレクトリが作成され、以下が出力されます:

- `results/documents/*.json` – 各 PDF についての抽出結果＋正規化された変数/方程式。
- `results/models/models.json` – ModelBuilder が生成した候補モデル一覧。
- `results.json` – 上記ファイルへのインデックス情報。
- `config_<元ファイル名>.yaml` – 実行時の設定ファイルコピー。
- `artifacts/` – `--save-stage-io` 指定時のみ、各工程の入出力ログ。

最終的な `WorkflowOutput` JSON は標準出力にも表示されます。

---

## 開発メモ

- Python 3.11 以上を想定（`pyproject.toml` 参照）。
- 依存関係は [uv](https://github.com/astral-sh/uv) で管理し、`pyproject.toml` に定義しています。
  `openai`, `PyYAML` など。

---

## 英語版

英語版の説明は [README.md](README.md) を参照してください。
