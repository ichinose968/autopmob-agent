import json
import typing as t
from dataclasses import dataclass, field

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import MetricWithLLM, SingleTurnMetric
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ==========================================
# 1. 出力スキーマとプロンプト定義
# ==========================================

class EvaluationScore(BaseModel):
    tp: int = Field(..., description="True Positives: 正解データと一致した抽出項目の数")
    fp: int = Field(..., description="False Positives: 正解データに存在しないが抽出された項目の数（過検出）")
    fn: int = Field(..., description="False Negatives: 正解データにあるが抽出されなかった項目の数（未検出）")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1: float = Field(..., description="F1 score")
    accuracy: float = Field(..., description="Accuracy score (TP / (TP + FP + FN))")
    reasoning: str = Field(..., description="評価の根拠。どの項目が一致し、どれが間違っていたかの詳細。")

# 変数(Variables)評価用のプロンプト
VARIABLE_EVAL_PROMPT = """
あなたは科学論文のデータ抽出を評価する専門家です。
「抽出された変数リスト(Prediction)」と「正解変数リスト(Ground Truth)」を比較してください。

比較ルール:
1. **Symbol**: 完全に一致するか、LaTeX記法の差異（例: "x(t)" と "x"、"\\alpha" と "alpha"）は許容してください。
2. **Definition**: 言葉が違っても意味が同じであれば一致とみなしてください（例: "Velocity" と "Speed"）。
3. **Description**: 補助的な情報として使用してください。Symbolが曖昧な場合、ここが一致していれば正解とします。
4. 正解データの1つの項目につき、最もマッチする抽出項目を1つだけ対応させてください（多対一は不可）。

Ground Truth (Variables):
{ground_truth}

Prediction (Variables):
{prediction}

結果をJSON形式で出力してください。tp, fp, fn, precision, recall, f1, accuracy, reasoningを含めてください。
"""

# 数式(Equations)評価用のプロンプト
EQUATION_EVAL_PROMPT = """
あなたは科学論文の数式抽出を評価する専門家です。
「抽出された数式リスト(Prediction)」と「正解数式リスト(Ground Truth)」を比較してください。

比較ルール:
1. **LaTeX**: 数学的に等価であれば一致とみなしてください。空白の違い、括弧の有無、同値な表現（例: "\\frac{{d}}{{dt}}x" と "\\dot{{x}}"）は許容します。
2. **Description**: 式番号（Eq. 1など）や説明文が一致しているかを確認してください。LaTeXが複雑で判断しにくい場合の強い根拠になります。
3. **Variables**: 数式内で使われている変数のリストです。これが一致していることも正誤判定の参考にしてください。

Ground Truth (Equations):
{ground_truth}

Prediction (Equations):
{prediction}

結果をJSON形式で出力してください。tp, fp, fn, precision, recall, f1, accuracy, reasoningを含めてください。
"""

# ==========================================
# 2. Ragasカスタムメトリクスの実装
# ==========================================

class ComponentExtractionMetric(MetricWithLLM, SingleTurnMetric):
    def __init__(self, target_type: str):
        """
        Args:
            target_type (str): "variables" または "equations"
        """
        super().__init__()
        self.target_type = target_type.lower()
        self.name = f"extraction_{self.target_type}_f1"  # メトリクス名
        
        # タイプに応じたプロンプトを選択
        if self.target_type == "variables":
            self.instruction_prompt = VARIABLE_EVAL_PROMPT
        elif self.target_type == "equations":
            self.instruction_prompt = EQUATION_EVAL_PROMPT
        else:
            raise ValueError("target_type must be 'variables' or 'equations'")

    async def _ascore(self, row: t.Dict, callbacks: t.Any = None) -> float:
        return 0.0
    
    def _single_turn_ascore(self, sample: t.Dict, callbacks: t.Any = None) -> float:
        """
        同期実行用のメソッド (Ragasの仕様上、実装が必須)
        ここでは非同期メソッドを呼び出す形で実装します。
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # すでにイベントループが走っている場合（Jupyterなど）の簡易的な対応
            return 0.0 
        else:
            return loop.run_until_complete(self._asingle_turn_ascore(sample, callbacks))

    async def _asingle_turn_ascore(self, sample: t.Dict, callbacks: t.Any = None) -> float:
        """
        1行（1ドキュメント）ごとの評価を実行
        """
        prediction_json = sample.get("prediction_data")
        ground_truth_json = sample.get("ground_truth_data")

        # LLMへの入力作成
        prompt = self.instruction_prompt.format(
            ground_truth=ground_truth_json,
            prediction=prediction_json
        )

        # LLM呼び出し（Structured Outputを使用）
        # RagasのLLMラッパーは純粋な文字列生成用なので、ここではLangChainの機能を使って
        # 構造化データを直接取得する形にします。
        try:
            # self.llm.llm は LangChainのChatOpenAIインスタンス
            structured_llm = self.llm.llm.with_structured_output(EvaluationScore)
            result: EvaluationScore = await structured_llm.ainvoke(prompt)
            
            # スコアだけでなく、理由もログに残したい場合、
            # ここで何らかの方法で保存するか、戻り値に工夫が必要ですが、
            # Ragasの仕様上、返すのは float (F1 score) です。
            # 推論理由(reasoning)を保持するために、一時的にオブジェクトにアタッチすることも可能ですが、
            # シンプルにF1を返します。
            return result.f1

        except Exception as e:
            print(f"Error in extraction metric: {e}")
            return 0.0

# ==========================================
# 3. 実行用ヘルパー関数
# ==========================================

@dataclass
class ExtractionEvalResult:
    summary: pd.DataFrame
    details: list[dict]

def run_extraction_evaluation(
    predictions: list[t.Any], 
    ground_truths: list[t.Any],
    target_type: str = "variables",
    model_name: str = "gpt-4o"  # 数式の評価には高い推論能力が必要なため4o推奨
) -> pd.DataFrame:
    """
    抽出結果の評価を実行し、詳細なDataFrameを返します。
    """
    
    # 1. データをRagas用Datasetに変換
    # 複雑なオブジェクトをJSON文字列化して渡します
    data_points = []
    for pred, gt in zip(predictions, ground_truths):
        # Pydanticモデルなら .model_dump()、辞書ならそのまま
        pred_data = pred.model_dump() if hasattr(pred, "model_dump") else pred
        
        # 特定のキー（variables または equations）を取り出す
        if target_type in pred_data:
            pred_list = pred_data[target_type]
        else:
            # すでにリストそのものが渡されている場合のフォールバック
            pred_list = pred_data

        if target_type in gt:
            gt_list = gt[target_type]
        else:
            gt_list = gt

        data_points.append({
            "prediction_data": json.dumps(pred_list, ensure_ascii=False, indent=2),
            "ground_truth_data": json.dumps(gt_list, ensure_ascii=False, indent=2)
        })

    dataset = Dataset.from_list(data_points)

    # 2. メトリクスのセットアップ
    openai_llm = ChatOpenAI(model=model_name)
    
    # Ragas用にラップする（Metricクラスにはこれを渡す）
    llm_wrapper = LangchainLLMWrapper(openai_llm)
    metric = ComponentExtractionMetric(target_type=target_type)
    metric.llm = llm_wrapper

    # 3. 評価実行（直接LLMをループさせる）
    # ragas.evaluate() を使うと集計されてしまうため、詳細（TP/FPなどの生データ）が欲しい場合は
    # Metricクラスのロジックを直接バッチ実行する方が柔軟です。
    
    results = []
    structured_llm = openai_llm.with_structured_output(EvaluationScore)
    print(f"Starting evaluation for {target_type} using {model_name}...")
    
    # 同期的に実行（必要に応じてasyncio化してください）
    for i, row in enumerate(data_points):
        prompt = metric.instruction_prompt.format(
            ground_truth=row["ground_truth_data"],
            prediction=row["prediction_data"]
        )
        score_data: EvaluationScore = structured_llm.invoke(prompt)
        
        results.append({
            "doc_index": i,
            "target_type": target_type,
            "precision": score_data.precision,
            "recall": score_data.recall,
            "f1": score_data.f1,
            "accuracy": score_data.accuracy,
            "tp": score_data.tp,
            "fp": score_data.fp,
            "fn": score_data.fn,
            "reasoning": score_data.reasoning
        })

    return pd.DataFrame(results)

# ==========================================
# 4. 使用例 (Main)
# ==========================================

if __name__ == "__main__":
    # ダミーデータ（あなたのコードにある構造を想定）
    
    # 予測データ (Prediction)
    mock_preds = [
        {
            "variables": [
                {"symbol": "t", "definition": "Time", "description": "Independent variable"},
                {"symbol": "x", "definition": "State", "description": "System state vector"} # 表記揺れ: x vs x(t)
            ],
            "equations": [
                {
                    "latex": "\\frac{d}{dt}x(t) = f(x,u)", 
                    "variables": ["x", "u"], 
                    "description": "State equation (1)"
                }
            ]
        }
    ]

    # 正解データ (Ground Truth)
    mock_gts = [
        {
            "variables": [
                {"symbol": "t", "definition": "Time", "description": "Time parameter"},
                {"symbol": "x(t)", "definition": "State vector", "description": "Full state"},
                {"symbol": "u(t)", "definition": "Input", "description": "Control input"} # 未検出 (FN)
            ],
            "equations": [
                {
                    "latex": "\\dot{x} = f(x, u)", # LaTeX表記違い (\dot{x} vs d/dt)
                    "variables": ["x(t)", "u(t)"],
                    "description": "System dynamics Eq. 1"
                }
            ]
        }
    ]

    # 変数の評価
    df_vars = run_extraction_evaluation(mock_preds, mock_gts, target_type="variables")
    print("\n=== Variables Evaluation ===")
    print(df_vars[["precision", "recall", "f1", "reasoning"]])

    # 数式の評価
    df_eqs = run_extraction_evaluation(mock_preds, mock_gts, target_type="equations")
    print("\n=== Equations Evaluation ===")
    print(df_eqs[["precision", "recall", "f1", "reasoning"]])