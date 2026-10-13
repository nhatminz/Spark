"""Mathematical answer and response-format rewards."""


def reward_function():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    configuration = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]

    def parse_gold(answer):
        gold = parse(
            answer,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if not gold:
            raise ValueError("Unparseable mathematical ground truth")
        return gold

    def evaluate(text, gold):
        answer = parse(
            text, extraction_mode="first_match", extraction_config=configuration
        )
        try:
            correct = float(verify(answer, gold))
        except (ValueError, TypeError, TimeoutError):
            correct = 0.0
        return correct + 0.2 * float(text.count("\n</think>\n") == 1)

    return parse_gold, evaluate
