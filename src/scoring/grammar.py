import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def grammar_score(text):
    matches = tool.check(text)
    error_count = len(matches)
    score = max(0, 1 - error_count / 10)
    return score, matches