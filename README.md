# 研究PDF解析パイプライン

このリポジトリは、研究論文などのPDFを画像化した上で、OpenAIの視覚対応モデルに渡し、論文構造化JSON（DocJSON）や変数・数式データベースを生成するためのサンプル実装です。生成したJSONは `jsonschema` でバリデーションし、再利用しやすい形で保存できます。

## ディレクトリ構成

- `src/` – DocJSON 抽出パイプライン（`main.py`・`pdf_to_images.py`・`classify_extract_vlm.py`・`schema_docjson.py`）
- `src_1/` – 変数DB・数式DB抽出パイプライン（`main.py`・`classify_extract_vlm.py` など）
- `tests/Benavides_and_Diwekar_2012.pdf` – 動作確認用のサンプル論文
- `doc.json`, `DB_variable.json`, `DB_equation.json` – サンプル出力

## 事前準備

1. Python 3.10 以上を推奨します。
2. 仮想環境を用意する場合:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. 依存関係をインストールします:
   ```bash
   pip install openai python-dotenv jsonschema pymupdf pillow
   ```

## 環境変数

- `OPENAI_API_KEY` – OpenAI APIキー（`.env` に記載して `python-dotenv` で読み込み可能）
- `OPENAI_MODEL` – 使用するモデルID（既定値は `o4-mini`）

ルート直下に `.env` を置くと `main.py` 実行時に自動で読み込まれます。

## 使い方

### DocJSON を生成する

```bash
python -m src.main tests/Benavides_and_Diwekar_2012.pdf --out doc.json --dpi 220
```

流れ: PDF → ページ毎のPNG → OpenAI Responses API → DocJSON → `jsonschema` で検証 → ファイルへ保存。

### 変数・数式データベースを生成する

```bash
python -m src_1.main tests/Benavides_and_Diwekar_2012.pdf --var-out DB_variable.json --eq-out DB_equation.json
```

各スクリプトは共通の `pdf_to_images.py` で画像化し、モデルに与えるプロンプトとスキーマを変えてJSONを取得します。結果はそれぞれ `src_1/schema_docjson.py` に定義されたスキーマで検証されます。

## 出力フォーマット

- DocJSON スキーマ: `src/schema_docjson.py`
- 変数DBスキーマ: `src_1/schema_docjson.py`

各スキーマは `Draft 2020-12` 準拠の JSON Schema で、`jsonschema.Draft202012Validator` を用いて検証しています。必要に応じてスキーマを編集すれば、抽出結果の構造や必須項目を拡張できます。

## サンプルで試す

`tests/Benavides_and_Diwekar_2012.pdf` を入力に実行すると、既存の `doc.json` や `DB_variable.json` と近い出力を得られます。これらのファイルは期待値の参考例として活用できます。

## 開発メモ

- OpenAI API の利用には課金が発生するため、試行回数や解像度（`--dpi`）の設定に注意してください。
- ページ毎のPNGはメモリ上で処理しているため、長大なPDFでは追加の最適化が必要になる場合があります。
- モデル応答がスキーマに適合しない場合は、プロンプトやスキーマの調整、再試行ロジックの追加などで対処してください。
