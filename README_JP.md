## AutoPMoB Agent（日本語版）

AutoPMoB Agent は、技術系 PDF から数理モデル化に必要なナラティブ・変数・
方程式を抽出し、目的に応じた候補モデルを生成する LangChain + OpenAI
ワークフローです。各ステージの結果は Pydantic で検証され、最終的に
「整備済み知識ベース＋モデル候補」の JSON として得られます。

```
PDF群 ──► Extractor ──► DBOrganizer ──► ModelBuilder ──► WorkflowOutput
             │              │                │
             │              │                └─ 目的別モデル候補
             │              └─ 正規化済み変数/方程式グラフ
             └─ 文書サマリ + 生データ抽出結果
```

## 特長
- **エンドツーエンド自動化** – 複数 PDF から変数・方程式を抽出し、目的に沿った
  モデル案をまとめて生成。
- **プロンプト差し替えが容易** – 各ステージは `system` / `human` を持つ YAML
  テンプレートで管理し、`{objective}` や `{documents}` などのプレースホルダを
  差し込み。
- **構造化された出力** – すべて Pydantic モデルで検証されるため、常に同じ JSON
  形式を下流システムに渡せます。
- **トレーサビリティ** – オプションで各ステージの入出力を `artifacts/` に保存
  し、LLM 応答を後から確認可能。

英語版は [README.md](README.md) を参照してください。

---

## ディレクトリ構成

```
autopmob-agent/
├── configs/          # ワークフロー設定 YAML
├── data/             # PDF などの入力ファイル
├── prompts/          # ステージ別プロンプト (例: prompts/type_1/*)
├── results/          # 実行結果（自動生成）
├── scripts/
│   └── model_builder_agent.py  # メインスクリプト
├── README.md
└── README_JP.md
```

---

## 事前準備
- Python **3.11 以上**
- 依存管理ツール [uv](https://github.com/astral-sh/uv)（推奨）
- OpenAI API キー（`gpt-5` 互換モデルにアクセス可能であること）
- 解析対象の PDF

API キーは `.env` に設定してください:

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

依存パッケージのインストール:

```bash
uv sync
```

---

## 設定ファイル (configs/*.yaml)

すべての実行は YAML 設定ファイルで制御します。`configs/cstr-sample.yaml`
をコピーし、下記項目を編集してください。

| セクション | 必須 | 内容 |
|------------|------|------|
| `pdf` | ✅ | `{path, doc_id?}` のリスト。パスはリポジトリ相対または絶対。`doc_id` があると後続解析が追跡しやすくなります。 |
| `objective.description` | ✅ | モデル化の目的。 |
| `objective.input_variables` | 任意 | 想定する入力変数。 |
| `objective.output_variables` | 任意 | 管理したい出力変数。 |
| `objective.success_criteria` | 任意 | 追加の成功条件／期待値。 |
| `max_models` | 任意 (既定 3) | 生成するモデル候補数。 |
| `model_name` | 任意 | 利用する OpenAI モデル名（既定 `gpt-5`）。 |
| `temperature` | 任意 | LLM の temperature。 |
| `prompts.extractor` / `organizer` / `model_builder` | ✅ | `system` と `human` を含む YAML ファイルパス、または設定ファイル内のマッピング。 |

最小例:

```yaml
pdf:
  - path: data/example/paper.pdf
    doc_id: case-1

objective:
  description: >
    CSTR の熱・物質収支を捉える非線形モデルを構築する。
  input_variables: [feed_flowrate, coolant_flowrate]
  output_variables: [reactor_temperature]
  success_criteria: >
    条件の異なるモデル構造を最低 2 つ提示すること。

max_models: 2
model_name: gpt-5
temperature: 0.2

prompts:
  extractor: prompts/type_1/extractor.yaml
  organizer: prompts/type_1/organizer.yaml
  model_builder: prompts/type_1/model_builder.yaml
```

### プロンプト形式

各プロンプトファイルは下記のように `system` / `human` キーを持ちます。

```yaml
system: |-
  システムロールの指示...
human: |-
  {objective} や {documents} を差し込むテンプレート
```

必須プレースホルダ:

- **Extractor** – `{objective}`, `{doc_id}`, `{title}`（PDF は自動で添付されます）
- **DBOrganizer** – `{objective}`, `{documents}`
- **ModelBuilder** – `{objective}`, `{db_snapshot}`, `{num_models}`

設定ファイルに直接マッピングを書くことも、ファイル参照にすることも可能です。

---

## 実行方法

```bash
uv run python scripts/model_builder_agent.py \
  --config configs/cstr-sample.yaml \
  --run-name cstr_run_001 \
  --save-stage-io
```
uv run python scripts/model_builder_agent.py \
  --config configs/cstr-sample.yaml \
  --run_name cstr_run_001 \
  --save_stage_io

### 主な CLI 引数

| フラグ | 説明 |
|--------|------|
| `--config PATH` | **必須。** ワークフロー設定ファイル。 |
| `--results-dir PATH` | 実行結果を保存する親ディレクトリ（既定 `results/`）。 |
| `--run-name NAME` | 実行フォルダ名（未指定時はタイムスタンプ）。 |
| `--save-stage-io` | 指定すると `artifacts/` に各ステージの入出力を保存。 |

`WorkflowOutput` は標準出力にも JSON として表示されます。

---

## 出力アーティファクト

`results/<run_name>/` に以下が作成されます:

- `results.json` – `organized_corpus` と `models` を含む最終 JSON。
- `results/documents/*.json` – 各 PDF ごとのナラティブ、抽出変数・方程式、
  正規化済みエントリへの参照。
- `results/models/models.json` – 生成された候補モデル一覧。
- `config_<元ファイル名>.yaml` – 実行時の設定ファイルコピー。
- `artifacts/*.json` – `--save-stage-io` 指定時のみ、各ステージの入出力ログ。

プロンプト調整や PDF 差し替えによる差分を簡単に比較できます。

---

## 開発メモ

- 依存関係は `pyproject.toml` に記述。`uv sync` でセットアップするか、お好みの方法を使用してください。
- CLI ロジックは `scripts/model_builder_agent.py` に集約されており、
  `Extractor / DBOrganizer / ModelBuilder` クラスで構成されています。
- ログ出力は既定で INFO。デバッグが必要な場合は環境変数 `LOGLEVEL=DEBUG`
  を設定するか、標準の logging 設定を上書きしてください。
- 詳細な使用方法は `uv run python scripts/model_builder_agent.py --help`
  で確認できます。

---

## ローカライズ

英語版 README は [README.md](README.md) にあります。ドキュメント更新時は
両ファイルの内容をそろえてください。
