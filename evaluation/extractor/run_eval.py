import json
import pandas as pd
from pathlib import Path
from evaluation import run_extraction_evaluation

def main():
    # 実行スクリプト(run_eval.py)の親ディレクトリパスを取得
    base_dir = Path(__file__).resolve().parent
    
    # 【変更点1】出力先ディレクトリの設定
    # autopmob-agent/evaluation/extractor/result にフォルダを作成・指定
    result_dir = base_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 入力データのパス設定
    gt_path = base_dir / "ground_truth.json"
    pred_path = base_dir / "predictions.json"

    print(f"データを読み込んでいます... \n正解データソース: {gt_path}")
    
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)
            
        with open(pred_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    except FileNotFoundError as e:
        print(f"エラー: 指定されたファイルが存在しません -> {e}")
        return

    # データの整列 (IDキーに基づくソート処理)
    print("データ整合性を確認中（IDソート実行）...")
    ground_truths.sort(key=lambda x: x.get('doc_id', ''))
    predictions.sort(key=lambda x: x.get('document_id', ''))

    if len(ground_truths) != len(predictions):
        print(f"警告: データセット間の件数が不一致です (正解: {len(ground_truths)}, 予測: {len(predictions)})")

    # === 1. 変数 (Variables) の評価フェーズ ===
    print("\n=== Phase 1: Variables (変数) 評価実行 ===")
    df_vars = run_extraction_evaluation(
        predictions, 
        ground_truths, 
        target_type="variables",
        model_name="gpt-4o"
    )
    
    # 【変更点2】Accuracy を出力カラムに追加
    cols = ["doc_index", "accuracy", "precision", "recall", "f1", "reasoning"]
    print(df_vars[cols])
    
    # 結果保存処理
    vars_save_path = result_dir / "result_variables.csv"
    df_vars.to_csv(vars_save_path, index=False, encoding="utf-8_sig")
    print(f"ステータス: 完了。出力ファイル -> {vars_save_path}")

    # === 2. 数式 (Equations) の評価フェーズ ===
    print("\n=== Phase 2: Equations (数式) 評価実行 ===")
    df_eqs = run_extraction_evaluation(
        predictions, 
        ground_truths, 
        target_type="equations",
        model_name="gpt-4o"
    )
    
    # 【変更点2】Accuracy を出力カラムに追加
    print(df_eqs[cols])
    
    # 結果保存処理
    eqs_save_path = result_dir / "result_equations.csv"
    df_eqs.to_csv(eqs_save_path, index=False, encoding="utf-8_sig")
    print(f"ステータス: 完了。出力ファイル -> {eqs_save_path}")

if __name__ == "__main__":
    main()