import os
import re
from functools import lru_cache

from transformers import AutoTokenizer


def wikitext_detokenizer(doc):
    string = doc["page"]
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


@lru_cache(maxsize=4)
def _load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool,
):
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )


def _get_tokenizer():
    model_name_or_path = os.environ.get("FOUROVERSIX_WIKITEXT_TOKENIZER")
    if not model_name_or_path:
        raise RuntimeError(
            "FOUROVERSIX_WIKITEXT_TOKENIZER must be set to compute token_perplexity.",
        )

    trust_remote_code = (
        os.environ.get("FOUROVERSIX_WIKITEXT_TOKENIZER_TRUST_REMOTE_CODE", "0") == "1"
    )
    return _load_tokenizer(model_name_or_path, trust_remote_code)


def _count_tokens(string: str) -> int:
    return len(_get_tokenizer().encode(string))


def process_results(doc, results):
    (loglikelihood,) = results
    # IMPORTANT: wikitext counts number of words in *original doc before detokenization*
    detokenized = wikitext_detokenizer(doc)
    _words = len(re.split(r"\s+", doc["page"]))
    _bytes = len(doc["page"].encode("utf-8"))
    _tokens = _count_tokens(detokenized)
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
        "token_perplexity": (loglikelihood, _tokens),
    }
