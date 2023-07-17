import os
from pydantic import BaseModel, Field
from simpleaichat import AIChat
from simpleaichat.utils import wikipedia_search, wikipedia_search_lookup


class get_event_metadata(BaseModel):
    """Event information"""

    description: str = Field(description="Description of event")
    city: str = Field(description="City where event occured")
    year: int = Field(description="Year when event occured")
    month: str = Field(description="Month when event occured")


# This uses the Wikipedia Search API.
# Results from it are nondeterministic, your mileage will vary.
def search(query):
    """Search wikipedia for a particular topic, input should be a name or a small sentence"""
    wiki_matches = wikipedia_search(query, n=3)
    return {"context": ", ".join(wiki_matches), "titles": wiki_matches}


def lookup(query):
    """Lookup more information about a topic."""
    page = wikipedia_search_lookup(query, sentences=3)
    return page


params = {"temperature": 0.0, "max_tokens": 100}
model = os.getenv("OPENAI_DEPLOYMENT_NAME")  # azure deployment name
api_url = os.getenv("OPENAI_API_BASE")  # https://xyz.openai.azure.com/
api_key = os.getenv("OPENAI_API_KEY")  # 123...
api_version = os.getenv("OPENAI_API_VERSION")  # 2023-06-01-preview
ai = AIChat(
    console=False,
    params=params,
    model=model,
    api_type="azure",
    api_url=api_url,
    api_version=api_version,
)
print(ai("First iPhone announcement", output_schema=get_event_metadata))
# print(ai("What are fun San Francisco tourist attractions?", tools=[search, lookup]))
