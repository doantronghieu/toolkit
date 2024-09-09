# How-to Guides

## Prompting

Prompt engineering is a technique used in natural language processing (NLP) to improve the performance of the language model by providing them with more context and information about the task in hand. It involves creating prompts, which are short pieces of text that provide additional information or guidance to the model, such as the topic or genre of the text it will generate. By using prompts, the model can better understand what kind of output is expected and produce more accurate and relevant results. In Llama 2, the size of the context, in terms of the number of tokens, has doubled from 2048 to 4096.

## Crafting Effective Prompts

Crafting effective prompts is an important part of prompt engineering. Here are some tips for creating prompts that will help improve the performance of your language model:

- **Be clear and concise**: Your prompt should be easy to understand and provide enough information for the model to generate relevant output. Avoid using jargon or technical terms that may confuse the model.
- **Use specific examples**: Providing specific examples in your prompt can help the model better understand what kind of output is expected. For example, if you want the model to generate a story about a particular topic, include a few sentences about the setting, characters, and plot.
- **Vary the prompts**: Using different prompts can help the model learn more about the task at hand and produce more diverse and creative output. Try using different styles, tones, and formats to see how the model responds.
- **Test and refine**: Once you have created a set of prompts, test them out on the model to see how it performs. If the results are not as expected, try refining the prompts by adding more detail or adjusting the tone and style.
- **Use feedback**: Finally, use feedback from users or other sources to continually improve your prompts. This can help you identify areas where the model needs more guidance and make adjustments accordingly.

## Explicit Instructions

Detailed, explicit instructions produce better results than open-ended prompts: You can think about giving explicit instructions as using rules and restrictions to how Llama 2 responds to your prompt.

### Stylization

Explain this to me like a topic on a children's educational network show teaching elementary students.

### Example

I'm a software engineer using large language models for summarization. Summarize the following text in under 250 words:

Give your answer like an old timey private investigator hunting down a case step by step.

### Formatting

- Use bullet points.
- Return as a JSON object.
- Use less technical terms and help me apply it in my work in communications.

### Restrictions

- Only use academic papers.
- Never give sources older than 2020.
- If you don't know the answer, say that you don't know.

Here's an example of giving explicit instructions to give more specific results by limiting the responses to recently created sources:

```plaintext
Explain the latest advances in large language models to me.
# More likely to cite sources from 2017

Explain the latest advances in large language models to me. Always cite your sources. Never cite sources older than 2020.
# Gives more specific advances and only cites sources from 2020
```

## Prompting using Zero- and Few-Shot Learning

A shot is an example or demonstration of what type of prompt and response you expect from a large language model. This term originates from training computer vision models on photographs, where one shot was one example or instance that the model used to classify an image.

### Zero-Shot Prompting

Large language models like Meta Llama are capable of following instructions and producing responses without having previously seen an example of a task. Prompting without examples is called "zero-shot prompting".

```plaintext
Text: This was the best movie I've ever seen!
The sentiment of the text is:

Text: The director was trying too hard.
The sentiment of the text is:
```

### Few-Shot Prompting

Adding specific examples of your desired output generally results in more accurate, consistent output. This technique is called "few-shot prompting". In this example, the generated response follows our desired format that offers a more nuanced sentiment classifier that gives a positive, neutral, and negative response confidence percentage.

```plaintext
You are a sentiment classifier. For each message, give the percentage of positive/neutral/negative. Here are some samples:
Text: I liked it
Sentiment: 70% positive 30% neutral 0% negative
Text: It could be better
Sentiment: 0% positive 50% neutral 50% negative
Text: It's fine
Sentiment: 25% positive 50% neutral 25% negative

Text: I thought it was okay

Text: I loved it!

Text: Terrible service 0/10
```

## Role-Based Prompts

Creating prompts based on the role or perspective of the person or entity being addressed. This technique can be useful for generating more relevant and engaging responses from language models.

### Pros:

- **Improves relevance**: Role-based prompting helps the language model understand the role or perspective of the person or entity being addressed, which can lead to more relevant and engaging responses.
- **Increases accuracy**: Providing additional context about the role or perspective of the person or entity being addressed can help the language model avoid making mistakes or misunderstandings.

### Cons:

- **Requires effort**: Requires more effort to gather and provide the necessary information about the role or perspective of the person or entity being addressed.

### Example:

```plaintext
You are a virtual tour guide currently walking the tourists Eiffel Tower on a night tour. Describe Eiffel Tower to your audience that covers its history, number of people visiting each year, amount of time it takes to do a full tour and why do so many people visit this place each year.
```

## Chain of Thought Technique

Involves providing the language model with a series of prompts or questions to help guide its thinking and generate a more coherent and relevant response. This technique can be useful for generating more thoughtful and well-reasoned responses from language models.

### Pros:

- **Improves coherence**: Helps the language model think through a problem or question in a logical and structured way, which can lead to more coherent and relevant responses.
- **Increases depth**: Providing a series of prompts or questions can help the language model explore a topic more deeply and thoroughly, potentially leading to more insightful and informative responses.

### Cons:

- **Requires effort**: The chain of thought technique requires more effort to create and provide the necessary prompts or questions.

### Example:

```plaintext
You are a virtual tour guide from 1901. You have tourists visiting Eiffel Tower. Describe Eiffel Tower to your audience. Begin with
1. Why it was built
2. Then by how long it took them to build
3. Where were the materials sourced to build
4. Number of people it took to build
5. End it with the number of people visiting the Eiffel tour annually in the 1900's, the amount of time it completes a full tour and why so many people visit this place each year.
Make your tour funny by including 1 or 2 funny jokes at the end of the tour.
```

## Self-Consistency

LLMs are probabilistic, so even with Chain-of-Thought, a single generation might produce incorrect results. Self-Consistency introduces enhanced accuracy by selecting the most frequent answer from multiple generations (at the cost of higher compute):

```plaintext
John found that the average of 15 numbers is 40.
If 10 is added to each number then the mean of the numbers is?
Report the answer surrounded by three backticks, for example: ```123```
```

Running the above several times and taking the most commonly returned value for the answer would make use of the self-consistency approach.

## Retrieval-Augmented Generation

Common facts are generally available from today's large models out-of-the-box (i.e. using just the model weights). More specific data is unlikely to be available though.

### Example:

```plaintext
What is the capital of California?
# The capital of California is Sacramento...

What was the temperature in Menlo Park on December 12th, 2023?
# I'm just an AI, I don't have access to real-time or historical weather data...
```

Retrieval-Augmented Generation, or RAG, describes the practice of including information in the prompt that has been retrieved from an external database. It's an effective way to incorporate facts into your LLM application and is more affordable than fine-tuning which might also negatively impact the foundational model's capabilities.

### Example:

```plaintext
Given the following information about temperatures in Menlo Park:
2023-12-11 :  52 degrees Fahrenheit
2023-12-12 :  51 degrees Fahrenheit
2023-12-13 :  55 degrees Fahrenheit
What was the temperature in Menlo Park on 2023-12-12?

# Sure! The temperature in Menlo Park on 2023-12-12 was 51 degrees Fahrenheit.

What was the temperature in Menlo Park on 2023-07-18 ?

# Sorry, I don't have information about the temperature in Menlo Park on 2023-07-18. The information provided only includes temperatures for December 11th, 12th, and 13th of 2023.
```

## Program-Aided Language Models

LLMs, by nature, aren't great at performing calculations. While LLMs are bad at arithmetic, they're great for code generation. Program-Aided Language leverages this fact by instructing the LLM to write code to solve calculation tasks.

### Example:

```plaintext
Python code to calculate: ((-5 + 93 * 4 - 0) * (4^4 + -7 + 0 * 5))

Here is the Python code to calculate the given expression:

((-5 + 93 * 4 - 0) * (4**4 + -7 + 0 * 5))

# Calculate the expression inside the parentheses
a = -5 + 93 * 4 - 0
b = 4**4 + -7 + 0 * 5

# Multiply a and b
result = a * b

# Print the result
print(result)
```

Using the code directly provides the correct result.

## Limiting Extraneous Tokens

A common challenge is generating a response without extraneous tokens (e.g. "Sure! Here's more information on...").

By combining a role, rules and restrictions, explicit instructions, and an example, the model can be prompted to generate the desired response.

### Example:

```plaintext
You are a robot that only outputs JSON.
You reply in JSON format with the field 'zip_code'.
Example question: What is the zip code of the Empire State Building?
Example answer: {'zip_code': 10118}
Now here is my question: What is the zip code of Menlo Park?

# "{'zip_code': 94025}"
```

Using the code directly provides the correct result.

## Reduce Hallucinations

Meta’s Responsible Use Guide is a great resource to understand how best to prompt and address input/output risks of the language model. Refer to pages (14-17).

Here are some examples of how a language model might hallucinate and some strategies for fixing the issue:

### Example 1:

A language model is asked to generate a response to a question about a topic it has not been trained on. The language model may hallucinate information or make up facts that are not accurate or supported by evidence.

**Fix**: To fix this issue, you can provide the language model with more context or information about the topic to help it understand what is being asked and generate a more accurate response. You could also ask the language model to provide sources or evidence for any claims it makes to ensure that its responses are based on factual information.

### Example 2:

A language model is asked to generate a response to a question that requires a specific perspective or point of view. The language model may hallucinate information or make up facts that are not consistent with the desired perspective or point of view.

**Fix**: To fix this issue, you can provide the language model with additional information about the desired perspective or point of view, such as the goals, values, or beliefs of the person or entity being addressed. This can help the language model understand the context and generate a response that is more consistent with the desired perspective or point of view.

### Example 3:

A language model is asked to generate a response to a question that requires a specific tone or style. The language model may hallucinate information or make up facts that are not consistent with the desired tone or style.

**Fix**: To fix this issue, you can provide the language model with additional information about the desired tone or style, such as the audience or purpose of the communication. This can help the language model understand the context and generate a response that is more consistent with the desired tone or style.

Overall, the key to avoiding hallucination in language models is to provide them with clear and accurate information and context, and to carefully monitor their responses to ensure that they are consistent with your expectations and requirements.

## Prompting Techniques

### Explicit Instructions

Detailed, explicit instructions produce better results than open-ended prompts:

```plaintext
complete_and_print(prompt="Describe quantum physics in one short sentence of no more than 12 words")
```

You can think about giving explicit instructions as using rules and restrictions to how Llama 3 responds to your prompt.

### Stylization

Explain this to me like a topic on a children's educational network show teaching elementary students.

I'm a software engineer using large language models for summarization. Summarize the following text in under 250 words:

Give your answer like an old timey private investigator hunting down a case step by step.

### Formatting

- Use bullet points.
- Return as a JSON object.
- Use less technical terms and help me apply it in my work in communications.

### Restrictions

- Only use academic papers.
- Never give sources older than 2020.
- If you don't know the answer, say that you don't know.

Here's an example of giving explicit instructions to give more specific results by limiting the responses to recently created sources.

```plaintext
complete_and_print("Explain the latest advances in large language models to me.")
# More likely to cite sources from 2017

complete_and_print("Explain the latest advances in large language models to me. Always cite your sources. Never cite sources older than 2020.")
# Gives more specific advances and only cites sources from 2020
```

### Example Prompting using Zero- and Few-Shot Learning

A shot is an example or demonstration of what type of prompt and response you expect from a large language model. This term originates from training computer vision models on photographs, where one shot was one example or instance that the model used to classify an image (Fei-Fei et al. (2006)).

### Zero-Shot Prompting

Large language models like Llama 3 are unique because they are capable of following instructions and producing responses without having previously seen an example of a task. Prompting without examples is called "zero-shot prompting".

```plaintext
complete_and_print("Text: This was the best movie I've ever seen! \n The sentiment of the text is: ")
# Returns positive sentiment

complete_and_print("Text: The director was trying too hard. \n The sentiment of the text is: ")
# Returns negative sentiment
```

### Few-Shot Prompting

Adding specific examples of your desired output generally results in more accurate, consistent output. This technique is called "few-shot prompting".

In this example, the generated response follows our desired format that offers a more nuanced sentiment classifier that gives a positive, neutral, and negative response confidence percentage.

See also: Zhao et al. (2021), Liu et al. (2021), Su et al. (2022), Rubin et al. (2022).

```python
def sentiment(text):
    response = chat_completion(messages=[
        user("You are a sentiment classifier. For each message, give the percentage of positive/neutral/negative."),
        user("I liked it"),
        assistant("70% positive 30% neutral 0% negative"),
        user("It could be better"),
        assistant("0% positive 50% neutral 50% negative"),
        user("It's fine"),
        assistant("25% positive 50% neutral 25% negative"),
        user(text),
    ])
    return response

def print_sentiment(text):
    print(f'INPUT: {text}')
    print(sentiment(text))

print_sentiment("I thought it was okay")
# More likely to return a balanced mix of positive, neutral, and negative
print_sentiment("I loved it!")
# More likely to return 100% positive
print_sentiment("Terrible service 0/10")
# More likely to return 100% negative
```

### Role Prompting

Llama will often give more consistent responses when given a role (Kong et al. (2023)). Roles give context to the LLM on what type of answers are desired.

Let's use Llama 3 to create a more focused, technical response for a question around the pros and cons of using PyTorch.

```plaintext
complete_and_print("Explain the pros and cons of using PyTorch.")
# More likely to explain the pros and cons of PyTorch covers general areas like documentation, the PyTorch community, and mentions a steep learning curve

complete_and_print("Your role is a machine learning expert who gives highly technical advice to senior engineers who work with complicated datasets. Explain the pros and cons of using PyTorch.")
# Often results in more technical benefits and drawbacks that provide more technical details on how model layers
```

### Chain-of-Thought

Simply adding a phrase encouraging step-by-step thinking "significantly improves the ability of large language models to perform complex reasoning" (Wei et al. (2022)). This technique is called "CoT" or "Chain-of-Thought" prompting.

Llama 3.1 now reasons step-by-step naturally without the addition of the phrase. This section remains for completeness.

```plaintext
prompt = "Who lived longer, Mozart or Elvis?"

complete_and_print(prompt)
# Llama 2 would often give the incorrect answer of "Mozart"

complete_and_print(f"{prompt} Let's think through this carefully, step by step.")
# Gives the correct answer "Elvis"
```

### Self-Consistency

LLMs are probabilistic, so even with Chain-of-Thought, a single generation might produce incorrect results. Self-Consistency (Wang et al. (2022)) introduces enhanced accuracy by selecting the most frequent answer from multiple generations (at the cost of higher compute):

```python
import re
from statistics import mode

def gen_answer():
    response = completion(
        "John found that the average of 15 numbers is 40."
        "If 10 is added to each number then the mean of the numbers is?"
        "Report the answer surrounded by backticks (example: `123`)",
    )
    match = re.search(r'`(\d+)`', response)
    if match is None:
        return None
    return match.group(1)

answers = [gen_answer() for i in range(5)]

print(
    f"Answers: {answers}\n",
    f"Final answer: {mode(answers)}",
    )

# Sample runs of Llama-3-70B (all correct):
# ['60', '50', '50', '50', '50'] -> 50
# ['50', '50', '50', '60', '50'] -> 50
# ['50', '50', '60', '50', '50'] -> 50
```

### Retrieval-Augmented Generation

You'll probably want to use factual knowledge in your application. You can extract common facts from today's large models out-of-the-box (i.e. using just the model weights):

```plaintext
complete_and_print("What is the capital of the California?")
# Gives the correct answer "Sacramento"
```

However, more specific facts, or private information, cannot be reliably retrieved. The model will either declare it does not know or hallucinate an incorrect answer:

```plaintext
complete_and_print("What was the temperature in Menlo Park on December 12th, 2023?")
# "I'm just an AI, I don't have access to real-time weather data or historical weather records."

complete_and_print("What time is my dinner reservation on Saturday and what should I wear?")
# "I'm not able to access your personal information [..] I can provide some general guidance"
```

Retrieval-Augmented Generation, or RAG, describes the practice of including information in the prompt you've retrieved from an external database (Lewis et al. (2020)). It's an effective way to incorporate facts into your LLM application and is more affordable than fine-tuning which may be costly and negatively impact the foundational model's capabilities.

This could be as simple as a lookup table or as sophisticated as a vector database) containing all of your company's knowledge:

```python
MENLO_PARK_TEMPS = {
    "2023-12-11": "52 degrees Fahrenheit",
    "2023-12-12": "51 degrees Fahrenheit",
    "2023-12-13": "51 degrees Fahrenheit",
}

def prompt_with_rag(retrieved_info, question):
    complete_and_print(
        f"Given the following information: '{retrieved_info}', respond to: '{question}'"
    )

def ask_for_temperature(day):
    temp_on_day = MENLO_PARK_TEMPS.get(day) or "unknown temperature"
    prompt_with_rag(
        f"The temperature in Menlo Park was {temp_on_day} on {day}'",  # Retrieved fact
        f"What is the temperature in Menlo Park on {day}?",  # User question
    )

ask_for_temperature("2023-12-12")
# "Sure! The temperature in Menlo Park on 2023-12-12 was 51 degrees Fahrenheit."

ask_for_temperature("2023-07-18")
# "I'm not able to provide the temperature in Menlo Park on 2023-07-18 as the information provided states that the temperature was unknown."
```

### Program-Aided Language Models

LLMs, by nature, aren't great at performing calculations. Let's try:

(The correct answer is 91383.)

```plaintext
complete_and_print("""
Calculate the answer to the following math problem:

((-5 + 93 * 4 - 0) * (4^4 + -7 + 0 * 5))
""")
# Gives incorrect answers like 92448, 92648, 95463
```

Gao et al. (2022) introduced the concept of "Program-aided Language Models" (PAL). While LLMs are bad at arithmetic, they're great for code generation. PAL leverages this fact by instructing the LLM to write code to solve calculation tasks.

```plaintext
complete_and_print(
    """
    # Python code to calculate: ((-5 + 93 * 4 - 0) * (4^4 + -7 + 0 * 5))
    """,
)
# The following code was generated by Llama 3 70B:

result = ((-5 + 93 * 4 - 0) * (4**4 - 7 + 0 * 5))
print(result)
```

### Limiting Extraneous Tokens

A common struggle with Llama 2 is getting output without extraneous tokens (ex. "Sure! Here's more information on..."), even if explicit instructions are given to Llama 2 to be concise and no preamble. Llama 3.x can better follow instructions.

Check out this improvement that combines a role, rules and restrictions, explicit instructions, and an example:

```plaintext
complete_and_print(
    "Give me the zip code for Menlo Park in JSON format with the field 'zip_code'",
)
# Likely returns the JSON and also "Sure! Here's the JSON..."

complete_and_print(
    """
    You are a robot that only outputs JSON.
    You reply in JSON format with the field 'zip_code'.
    Example question: What is the zip code of the Empire State Building? Example answer: {'zip_code': 10118}
    Now here is my question: What is the zip code of Menlo Park?
    """,
)
# "{'zip_code': 94025}"
```