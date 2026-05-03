import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelPricing:
    input_per_million: float
    output_per_million: float
    batch_discount: float  # e.g. 0.5 = 50% cheaper in batch/delayed mode

    def cost(self, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
        multiplier = (1 - self.batch_discount) if batch else 1.0
        return (
            input_tokens / 1_000_000 * self.input_per_million
            + output_tokens / 1_000_000 * self.output_per_million
        ) * multiplier


@dataclass
class Config:
    # Doubleword / DeepSeek
    doubleword_api_key: str
    doubleword_base_url: str
    deepseek_model: str
    deepseek_max_context: int

    # Gemini
    gemini_api_key: str
    gemini_flash_model: str
    gemini_pro_model: str
    gemini_max_context: int

    # Eval settings
    target_context_tokens: int
    results_dir: str

    # Batch / delayed mode
    use_doubleword_batch: bool      # use OpenAI batch API with completion_window="1h"
    use_gemini_batch: bool          # use google-genai batch API if available
    batch_poll_interval_s: int      # seconds between batch status polls

    # Pricing (USD per million tokens)
    deepseek_pricing: ModelPricing
    gemini_flash_pricing: ModelPricing
    gemini_pro_pricing: ModelPricing


def load_config() -> Config:
    missing = [k for k in ("DOUBLEWORD_API_KEY", "GEMINI_API_KEY") if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your API keys."
        )

    return Config(
        doubleword_api_key=os.environ["DOUBLEWORD_API_KEY"],
        doubleword_base_url=os.environ.get("DOUBLEWORD_BASE_URL", "https://api.doubleword.ai/v1"),
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        deepseek_max_context=int(os.environ.get("DEEPSEEK_MAX_CONTEXT", "65536")),

        gemini_api_key=os.environ["GEMINI_API_KEY"],
        gemini_flash_model=os.environ.get("GEMINI_FLASH_MODEL", "gemini-3-flash-preview"),
        gemini_pro_model=os.environ.get("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview"),
        gemini_max_context=int(os.environ.get("GEMINI_MAX_CONTEXT", "800000")),

        target_context_tokens=int(os.environ.get("TARGET_CONTEXT_TOKENS", "800000")),
        results_dir=os.environ.get("RESULTS_DIR", "results"),

        # Batch mode: Doubleword delayed ON by default, Gemini batch optional
        use_doubleword_batch=os.environ.get("USE_DOUBLEWORD_BATCH", "true").lower() == "true",
        use_gemini_batch=os.environ.get("USE_GEMINI_BATCH", "false").lower() == "true",
        batch_poll_interval_s=int(os.environ.get("BATCH_POLL_INTERVAL_S", "30")),

        # Pricing — update these to match current published rates
        # DeepSeek-V4-Flash via Doubleword (estimated, verify at doubleword.ai)
        deepseek_pricing=ModelPricing(
            input_per_million=float(os.environ.get("DEEPSEEK_INPUT_PRICE", "0.27")),
            output_per_million=float(os.environ.get("DEEPSEEK_OUTPUT_PRICE", "1.10")),
            batch_discount=float(os.environ.get("DEEPSEEK_BATCH_DISCOUNT", "0.50")),
        ),
        # Gemini 3 Flash (estimated, verify at ai.google.dev/pricing)
        gemini_flash_pricing=ModelPricing(
            input_per_million=float(os.environ.get("GEMINI_FLASH_INPUT_PRICE", "0.075")),
            output_per_million=float(os.environ.get("GEMINI_FLASH_OUTPUT_PRICE", "0.30")),
            batch_discount=float(os.environ.get("GEMINI_FLASH_BATCH_DISCOUNT", "0.50")),
        ),
        # Gemini 3 Pro (judge only — not run in batch mode)
        gemini_pro_pricing=ModelPricing(
            input_per_million=float(os.environ.get("GEMINI_PRO_INPUT_PRICE", "1.25")),
            output_per_million=float(os.environ.get("GEMINI_PRO_OUTPUT_PRICE", "5.00")),
            batch_discount=0.0,
        ),
    )
