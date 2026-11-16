"""Native Gemini adapter that omits controls removed by Gemini 3.8 Flash."""

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


class PatchedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """Preserve native SDK behavior while removing obsolete request defaults."""

    def _prepare_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        request = super()._prepare_request(*args, **kwargs)
        if self.model.removeprefix("models/") == "gemini-3.8-flash":
            # https://aistudio.google.com/docs/whats-new-gemini-3.8#migrate-to-gemini-3-8-flash
            config = request["config"]
            if config.candidate_count not in (None, 1):
                raise ValueError("Gemini 3.8 Flash supports only one candidate per request.")
            request["config"] = config.model_copy(update={"candidate_count": None, "temperature": None, "top_p": None, "top_k": None})
        return request
