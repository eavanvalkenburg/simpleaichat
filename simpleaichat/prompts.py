TOOL_PROMPT = """From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response to the user's last message.
- If no tool is appropriate, ONLY reply with \"0\".

{tools}"""

TOOL_INPUT_PROMPT = """You have to specify the input for a function that does: {tool}. 
- Reply only with a concise answer that is appropriate for the tool and nothing else
- If no input is required, return `None` and nothing else."""
