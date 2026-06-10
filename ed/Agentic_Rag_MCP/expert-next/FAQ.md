Please read this webpage:
https://edwarddonner.com/faq

And create a file `knowledge/faq.jsonl` that contains the contents of the FAQ. If the file already exists, update it as needed with any new or changed questions.
The jsonl should have this format on each line:

```
{"faq": 1, "question": "a clearly stated rewrite of the question, suitable for an agent to understand the question clearly in order to answer a user", "answer": "the answer to the question with the original wording but using Markdown formatting; don't include any diagrams, but update the wording if needed to explain what the diagram was conveying instead of the diagram"}
```

This file will be used by an Agent that is trying to answer user questions. The question should be rewritten to maximize the chance that the Agent understands it. The answer should as much as possible preserve the original content, in Markdown format.