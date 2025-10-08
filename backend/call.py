import logging

from utils.llm_caller import LLM
from .backend_utils import compile_prompt_to_md
from backend.backend_utils import PromptType
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

def r1_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    logger.info(f"prompt: {prompt}", extra={"verbose": True})
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Get response with metrics
    result = llm.stream_complete(
        prompt,
        return_metrics=True,
        **model_kwargs
    )
    
    # Extract text and metrics
    response = result["text"] if isinstance(result, dict) else result
    res = response[response.find("</think>")+8:]
    
    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"response without think:\n{res}", extra={"verbose": True})
    
    # Log metrics WITHOUT verbose flag so it appears in both ml-master.log and ml-master.verbose.log
    if isinstance(result, dict):
        logger.info(
            f"[METRICS] Input tokens: {result['input_tokens']}, "
            f"Output tokens: {result['output_tokens']}, "
            f"Total tokens: {result['total_tokens']}, "
            f"Response time: {result['response_time']:.2f}s, "
            f"Tokens/sec: {result['output_tokens']/result['response_time']:.1f}"
        )
    
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res